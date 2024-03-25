sensitive_info_patterns = [
    # Personal Information
    [{"LOWER": "full"}, {"LOWER": "name"}],
    [{"LOWER": "date"}, {"LOWER": "of"}, {"LOWER": "birth"}],
    [{"LOWER": "address"}],
    [{"LOWER": "phone"}, {"LOWER": "number"}],
    [{"LOWER": "email"}, {"LOWER": "address"}],
    [{"LOWER": "account"}, {"LOWER": "details"}],
    [{"LOWER": "account"}, {"LOWER": "information"}],
    [{"LOWER": "your"}, {"LOWER": "account"}],
    [{"LOWER": "social"}, {"LOWER": "security"}, {"LOWER": "number"}, {"IS_PUNCT": True, "OP": "?"}],
    [{"LOWER": "password"}],

    # Financial Information
    [{"LOWER": "bank"}, {"LOWER": "account"}],
    [{"LOWER": "routing"}, {"LOWER": "number"}],
    [{"LOWER": "credit"}, {"LOWER": "card"}],
    [{"LOWER": "debit"}, {"LOWER": "card"}],
    [{"LOWER": "card"}, {"LOWER": "expiry"}, {"LOWER": "date"}],
    [{"LOWER": "cvv"}],
    [{"LOWER": "pin"}],

    # Security Information
    [{"LOWER": "security"}, {"LOWER": "question"}, {"LOWER": "answer"}],
    [{"LOWER": "verification"}, {"LOWER": "code"}],
    [{"LOWER": "login"}, {"LOWER": "details"}],
    [{"LOWER": "username"}, {"LOWER": "and"}, {"LOWER": "password"}],
    [{"LOWER": "authenticate"}, {"LOWER": "your"}, {"LOWER": "account"}],

    # Urgent Requests
    [{"LOWER": "confirm"}, {"LOWER": "identity"}],
    [{"LOWER": "verify"}, {"LOWER": "account"}],
    [{"LOWER": "verify"}, {"LOWER": "information"}],
    [{"LOWER": "update"}, {"LOWER": "account"}],
    [{"LOWER": "update"}, {"LOWER": "payment"}, {"LOWER": "info"}],
    [{"LOWER": "immediate"}, {"LOWER": "action"}, {"LOWER": "required"}],

    # Phrases that might be used to trick users into providing information
    [{"LOWER": "unlock"}, {"LOWER": "your"}, {"LOWER": "account"}],
    [{"LOWER": "secure"}, {"LOWER": "your"}, {"LOWER": "account"}],
    [{"LOWER": "account"}, {"LOWER": "suspension"}, {"LOWER": "notice"}],
    [{"LOWER": "compliance"}, {"LOWER": "verification"}],
]

urgency_keywords = [
    "immediate", "now", "urgent", "important", "action required",
    "as soon as possible", "asap", "right away", "at your earliest convenience",
    "prompt", "don't delay", "hurry", "quick", "immediately",
    "deadline", "limited time", "expire", "final notice", "last chance",
    "today only", "time sensitive", "time-sensitive", "critical",
    "rush", "warning", "alert", "instant", "immediate action",
    "requires your attention", "requires immediate attention",
    "don’t miss out", "before it’s too late"
]

generic_greeting_patterns = [
    # Simple greetings
    [{"LOWER": "hi"}], [{"LOWER": "hello"}], [{"LOWER": "hey"}],
    [{"LOWER": "greetings"}], [{"LOWER": "dear"}, {"IS_ALPHA": True, "OP": "+"}],
    [{"LOWER": "good"}, {"LOWER": {"IN": ["morning", "afternoon", "evening", "day"]}}],

    # Formal greetings
    [{"LOWER": "dear"}, {"LOWER": "sir"}], [{"LOWER": "dear"}, {"LOWER": "madam"}],
    [{"LOWER": "to"}, {"LOWER": "whom"}, {"LOWER": "it"}, {"LOWER": "may"}, {"LOWER": "concern"}],

    # Greetings with titles and names
    [{"LOWER": "mr."}, {"IS_ALPHA": True}], [{"LOWER": "ms."}, {"IS_ALPHA": True}],
    [{"LOWER": "mrs."}, {"IS_ALPHA": True}], [{"LOWER": "dr."}, {"IS_ALPHA": True}],

    # Multilingual greetings
    [{"LOWER": {"IN": ["hola", "bonjour", "hallo", "ciao", "こんにちは", "안녕하세요"]}}],

    # Email-specific greetings
    [{"LOWER": {"IN": ["dear", "attention", "hello", "hi"]}}, {"IS_PUNCT": True, "OP": "?"},
     {"IS_SPACE": True, "OP": "*"}, {"LOWER": "all"}],
    [{"LOWER": {"IN": ["dear", "attention", "hello", "hi"]}}, {"LOWER": "team"}],
    [{"LOWER": {"IN": ["dear", "attention", "hello", "hi"]}}, {"LOWER": "valued"}, {"LOWER": "customer"}],
    [{"LOWER": {"IN": ["dear", "attention", "hello", "hi"]}}, {"LOWER": "valued"}, {"LOWER": "client"}],

    # Greetings that may appear in phishing
    [{"LOWER": {"IN": ["urgent", "important", "immediate"]}}, {"IS_SPACE": True, "OP": "*"}, {"LOWER": "attention"}],
    [{"LOWER": "attention"}, {"LOWER": {"IN": ["required", "needed"]}}],
]