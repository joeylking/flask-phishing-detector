import email
import mailbox
import re
from email.message import EmailMessage
from email.utils import getaddresses

import readability as readability
import spacy
from bs4 import BeautifulSoup
from spacy.matcher import Matcher
from spellchecker import SpellChecker

from . import matcher_patterns


class FeatureExtractor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_md')
        self.matcher = Matcher(self.nlp.vocab)
        self.matcher.add("SENSITIVE_INFO_REQUEST", matcher_patterns.sensitive_info_patterns)
        self.matcher.add("URGENCY", [[{"LOWER": keyword}] for keyword in matcher_patterns.urgency_keywords])
        self.matcher.add("GREETINGS", matcher_patterns.generic_greeting_patterns)
        self.url_pattern = re.compile(r'https?://[^\s]+')
        self.script_pattern = re.compile(r'<\w+\s+src\s*=\s*["\']https?://[^\s"\']+\.js["\']', re.IGNORECASE)

    def extract(self, email):
        features = {}

        email_message, email_body, email_subject, email_recipients, soup = self.clean_and_parse_email(email)
        html_content, plain_text_content = self.extract_html_and_plain_text(email_message)
        body_doc = self.nlp(email_body)
        body_matches = self.matcher(body_doc)

        features['has_sensitive_info'], \
            features['sensitive_info_phrases'] = self.contains_sensitive_info_request(body_doc, body_matches)
        features['has_imperatives'], features['imperative_phrases'] = self.contains_imperatives(body_doc)
        features['has_urgency'], features['urgency_phrases'] = self.contains_urgency(body_matches, body_doc)
        features['spelling_errors'] = self.contains_spelling_errors(email_body)
        features['has_generic_greeting'] = self.contains_generic_greeting(body_matches)
        features['subject_length'] = len(str(email_subject))
        features['body_length'] = len(email_body)
        features['has_html'] = bool(html_content)
        features['has_javascript'] = self.contains_javascript(soup, html_content)
        features['has_links'], \
            features['links_count'] = self.extract_link_features(soup, html_content, plain_text_content)
        features['num_of_recipients'] = len(email_recipients)
        features['has_attachments'], features['attachments_count'] = self.extract_attachment_features(email_message)

        if plain_text_content.strip():
            features['readability_score'] = \
                readability.getmeasures(plain_text_content, lang='en')['readability grades'][
                    'FleschReadingEase']
        else:
            features['readability_score'] = None
        return features

    def clean_and_parse_email(self, input_data):
        # Check if input_data is an mbox message or a standard email message
        if isinstance(input_data, (mailbox.mboxMessage, EmailMessage, email.message.Message)):
            # Directly use the email object as it is already a Message object
            email_message = input_data
        elif isinstance(input_data, str):
            # Directly parse the string to EmailMessage
            email_message = email.message_from_string(input_data)
        else:
            raise ValueError("Unsupported email input type.")

        # Extracting recipients
        to_recipients = email_message.get_all('To', [])
        cc_recipients = email_message.get_all('Cc', [])
        bcc_recipients = email_message.get_all('Bcc', [])
        all_recipients = set(getaddresses(to_recipients + cc_recipients + bcc_recipients))

        # Extract the subject and body
        subject = email_message['Subject']
        body = ""
        if email_message.is_multipart():
            for part in email_message.walk():
                if part.get_content_type() in ["text/plain", "text/html"]:
                    try:
                        charset = part.get_content_charset() or 'utf-8'
                        body = part.get_payload(decode=True).decode(charset, errors='ignore')
                        break
                    except LookupError:
                        print(f"Unknown encoding {charset}, using 'utf-8' as fallback.")
                        body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        break
        else:
            try:
                charset = email_message.get_content_charset() or 'utf-8'
                body = email_message.get_payload(decode=True).decode(charset, errors='ignore')
            except LookupError:
                print(f"Unknown encoding {charset}, using 'utf-8' as fallback.")
                body = email_message.get_payload(decode=True).decode('utf-8', errors='ignore')

        # Clean HTML from the body if present
        soup = BeautifulSoup(body, 'html.parser')
        clean_body = soup.get_text(separator=' ')

        return email_message, clean_body, subject, all_recipients, soup

    def decode_payload(self, part):
        try:
            charset = part.get_content_charset() or 'utf-8'
            return part.get_payload(decode=True).decode(charset, 'ignore')
        except LookupError:
            return part.get_payload(decode=True).decode('utf-8', 'ignore')

    def extract_html_and_plain_text(self, email_message):
        html_content = ""
        plain_text_content = ""

        if email_message.is_multipart():
            for part in email_message.walk():
                if part.get_content_type() == 'text/html':
                    html_content += self.decode_payload(part)
                elif part.get_content_type() == 'text/plain':
                    plain_text_content += self.decode_payload(part)
        else:
            charset = email_message.get_content_charset() or 'utf-8'
            try:
                if email_message.get_content_type() == 'text/html':
                    html_content = email_message.get_payload(decode=True).decode(charset, 'ignore')
                elif email_message.get_content_type() == 'text/plain':
                    plain_text_content = email_message.get_payload(decode=True).decode(charset, 'ignore')
            except LookupError:
                html_content = email_message.get_payload(decode=True).decode('utf-8', 'ignore')

        return html_content, plain_text_content

    def contains_sensitive_info_request(self, doc, matches):
        # Retrieve the match ID for 'SENSITIVE_INFO_REQUEST' to filter matches
        sensitive_info_request_id = self.nlp.vocab.strings['SENSITIVE_INFO_REQUEST']

        # Initialize an empty list to hold the text of the matched spans
        sensitive_info_matches = []

        # Check if any matches belong to 'SENSITIVE_INFO_REQUEST'
        for match_id, start, end in matches:
            if match_id == sensitive_info_request_id:
                # Add the matched span text to the list
                sensitive_info_matches.append(doc[start:end].text)

        # Return True if there are sensitive info request matches found, and the list of matched spans
        return len(sensitive_info_matches) > 0, sensitive_info_matches

    def contains_imperatives(self, doc):
        imperatives = []
        for sent in doc.sents:
            verb = sent.root
            # Checking verb form and excluding sentences with explicit non-"you" subjects
            if verb.pos_ == "VERB" and verb.tag_ == "VB" and not any(
                    token.dep_ == "nsubj" and token.text.lower() != "you" for token in sent):
                # Further filter out sentences with modal verbs as roots, though this may exclude some imperatives
                if not verb.text.lower() in ["can", "could", "may", "might", "must", "shall", "should", "will",
                                             "would"]:
                    # Check if there are any children that could indicate an imperative structure
                    if any(child.dep_ in ("dobj", "xcomp", "advmod", "npadvmod") for child in verb.children):
                        imperatives.append(sent.text)
        return len(imperatives) > 0, imperatives

    def contains_urgency(self, matches, doc):
        urgency_matches = []

        # Iterate through matches to extract the text manually
        for match_id, start, end in matches:
            span = doc[start:end]  # Get the span from the doc
            if self.nlp.vocab.strings[match_id] == "URGENCY":
                urgency_matches.append(span.text)
        return len(urgency_matches) > 0, urgency_matches

    def contains_spelling_errors(self, text):
        # Initialize the spell checker
        spell = SpellChecker()

        # Tokenize the text into words
        words = re.findall(r'\b[a-z]+\b',
                           text.lower())  # This regex will match words and ignore punctuation and numbers

        # Find words that are misspelled
        misspelled = spell.unknown(words)

        return misspelled

    def contains_generic_greeting(self, matches):
        # Retrieve the match ID for "GREETINGS" patterns
        greetings_id = self.nlp.vocab.strings["GREETINGS"]

        # Check if any matches belong to 'GREETINGS'
        for match_id, start, end in matches:
            if match_id == greetings_id:
                return True  # A greeting match was found

        return False  # No greeting matches found

    def contains_javascript(self, soup, html_content):
        # Check for standard script tags
        if soup.find('script') is not None:
            return True

        # Check for obfuscated or malformed script tags
        if self.script_pattern.search(html_content):
            return True

        # Check for JavaScript event handlers
        for tag in soup.find_all(True):
            if any(attr in tag.attrs for attr in ['onload', 'onclick', 'onerror', 'onmouseover']):
                return True

        return False

    def extract_link_features(self, soup, html_content, plain_text_content):
        links_count = 0

        # Use BeautifulSoup to parse HTML content and find <a> tags
        if html_content:
            links_count += len(soup.find_all('a', href=True))

        plain_text_links = self.url_pattern.findall(plain_text_content)
        links_count += len(plain_text_links)

        # Determine if links are present and count them
        has_links = links_count > 0

        return has_links, links_count
    def extract_attachment_features(self, email_message):
        attachments_count = 0
        # Check if the email is multipart (attachments are in separate parts)
        if email_message.is_multipart():
            for part in email_message.walk():
                # The Content-Disposition header can be 'attachment' or 'inline' (for attachments shown directly in the
                # email body)
                if part.get_content_maintype() != 'multipart' and part.get("Content-Disposition") is not None:
                    attachments_count += 1

        return attachments_count > 0, attachments_count

    def extract_features_with_logging(self, index, dataset, email):
        print(f"Extracting features from email #{index + 1} of {dataset} dataset")
        return self.extract(email)
