import os
from dotenv import load_dotenv
from composio import Composio

load_dotenv(".env.local")

composio = Composio(
    api_key=os.getenv("COMPOSIO_API_KEY"),
    toolkit_versions={"gmail": "20260702_01"},
)

USER_ID = "cameron_test_trimmed"

result = composio.tools.execute(
    slug="GMAIL_CREATE_EMAIL_DRAFT",
    user_id=USER_ID,
    arguments={
        "recipient_email": "camdowdsonarai@gmail.com",
        "subject": "Composio proof of life",
        "body": "If you're reading this in your Drafts folder, Composio works.",
    },
)
print(result)
