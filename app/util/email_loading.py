import mailbox
import os
import random


def load_emails_from_directory(directory_path, emails):
    """
    Recursively load emails from a directory and its subdirectories.

    Args:
    - directory_path: The path to the directory to load emails from.
    - emails: A list to append the loaded email contents to.
    """
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isdir(item_path):
            # If the item is a directory, recurse into it
            load_emails_from_directory(item_path, emails)
        else:
            # If the item is a file, assume it's an email and try to read it
            try:
                with open(item_path, 'r', encoding='latin1') as email_file:
                    email_content = email_file.read()
                    emails.append(email_content)
                    # print(f"Email added from {directory_path} : {item_path}")
            except Exception as e:
                print(f"Error reading {item_path}: {e}")


def load_enron_dataset(maildir_root, folders_to_include=None, sample_fraction=0.5):
    """
    Load emails from the Enron dataset, navigating through subdirectories recursively.

    Args:
    - maildir_root: The root directory of the Enron maildir dataset.
    - folders_to_include: A list of folder names to include (e.g., ['inbox', 'sent_mail']). If None, all folders are
    included.
    - sample_fraction: The fraction of emails to sample from each folder (0.5 for half).
    Returns:
    - A list of email contents.
    """
    emails = []

    # Iterate through all user directories
    for person_dir in os.listdir(maildir_root):
        person_path = os.path.join(maildir_root, person_dir)
        if os.path.isdir(person_path):
            # If folders_to_include is specified, only look into those folders
            if folders_to_include:
                for folder_name in folders_to_include:
                    specific_path = os.path.join(person_path, folder_name)
                    if os.path.exists(specific_path):
                        load_emails_from_directory(specific_path, emails)
            else:
                # If no specific folders are specified, recursively load emails from all directories
                load_emails_from_directory(person_path, emails)
    print(f"Loaded {len(emails)} emails from specified folders.")
    # Sample a subset of emails randomly
    sampled_emails = random.sample(emails, int(len(emails) * sample_fraction))

    print(f"Loaded {len(sampled_emails)} emails from specified folders.")
    return sampled_emails


def load_nazario_phishing_dataset(nazario_dir,sample_fraction=0.5):
    print("Loading phishing dataset from:", nazario_dir)
    phishing_emails = []
    for root, dirs, files in os.walk(nazario_dir):
        for file in files:
            file_path = os.path.join(root, file)
            print("Processing file:", file_path)
            if file.endswith('.mbox'):
                try:
                    mbox = mailbox.mbox(file_path)
                    for message in mbox:
                        phishing_emails.append(message)
                except UnicodeDecodeError as e:
                    print(f"Unicode decode error in file {file_path}: {e}")
            else:
                print("not mbox file")
    print(f"Loaded {len(phishing_emails)} emails from specified folders.")

    # Sample a subset of emails randomly
    sampled_emails = random.sample(phishing_emails, int(len(phishing_emails) * sample_fraction))
    return sampled_emails


def safe_print_email(email_message):
    try:
        # Attempt to print the email as a string, assuming it's correctly encoded
        print(email_message.as_string())
    except UnicodeEncodeError:
        # If an error occurs, handle it by encoding the payload directly
        if email_message.is_multipart():
            for part in email_message.walk():
                charset = part.get_content_charset() or 'utf-8'
                try:
                    print(part.get_payload(decode=True).decode(charset, errors='replace'))
                except UnicodeError:
                    print(part.get_payload(decode=True).decode('utf-8', errors='replace'))
        else:
            charset = email_message.get_content_charset() or 'utf-8'
            try:
                print(email_message.get_payload(decode=True).decode(charset, errors='replace'))
            except UnicodeError:
                print(email_message.get_payload(decode=True).decode('utf-8', errors='replace'))