import pandas as pd
import sqlite3
import streamlit as st
from openai import OpenAI
import re

# Set page configuration
st.set_page_config(
    page_title="Question Answering System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Create custom CSS for better styling
st.markdown(
    """
<style>
    /* Main color scheme based on the reference image */
    :root {
        --main-blue: #0F75BC;
        --light-blue: #87CEEB;
        --accent-green: #12A89D;
        --light-bg: #f0f7fa;
        --button-hover: #0a5a8e;
    }

    /* Apply RTL direction globally */
    body {
        direction: rtl;
    }

    /* Ensure main app container respects RTL */
    .main .block-container {
         /* Adjust padding for RTL if necessary, often not needed */
         /* padding-left: 5rem; */
         /* padding-right: 5rem; */
    }

    /* Custom header styling - ensure content flows RTL */
    .main-header {
        background-color: var(--main-blue);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        direction: rtl; /* Explicitly set direction */
    }

    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        text-align: right; /* Align header text right */
    }

    /* Card styling for content sections */
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        direction: rtl; /* Already set, good */
        text-align: right; /* Default text align for cards */
    }

    /* Button styling */
    .stButton > button {
        background-color: var(--main-blue);
        color: white;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
        /* Ensure button text aligns center or as desired */
        text-align: center;
    }

    .stButton > button:hover {
        background-color: var(--button-hover);
        transform: translateY(-2px);
    }

    /* Input field styling - ensure text starts from the right */
    .stTextInput input, .stTextArea textarea {
        border-radius: 20px;
        padding: 1rem;
        border: 2px solid #eaeaea;
        direction: rtl; /* Ensure input direction is RTL */
        text-align: right; /* Align typed text and placeholder to the right */
    }

    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--accent-green);
        box-shadow: 0 0 0 0.2rem rgba(18, 168, 157, 0.25);
    }

    /* Expander styling - Header text align */
    .streamlit-expanderHeader {
        background-color: var(--light-bg);
        border-radius: 10px;
        color: var(--main-blue);
        font-weight: bold;
        text-align: right; /* Align expander header text */
    }
    /* Align the expander arrow icon to the left in RTL */
    .streamlit-expanderHeader > svg {
        order: -1; /* Moves the icon visually to the left */
        margin-right: 0;
        margin-left: 0.5rem; /* Add some space */
    }


    /* Tab styling - Ensure tabs flow RTL */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        /* Tabs should naturally flow RTL due to body direction */
    }

    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        border-radius: 5px 5px 0 0;
        padding: 0 1rem;
        background-color: #f1f1f1;
        text-align: right; /* Align tab text right */
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--main-blue);
        color: white;
    }

    /* Language toggle button styling (if re-enabled) */
    .language-toggle {
        display: flex;
        /* justify-content: flex-end; Should be flex-start in RTL */
        justify-content: flex-start;
        margin-bottom: 1rem;
    }
    /* Ensure radio buttons align right */
     .stRadio [role="radiogroup"] {
        align-items: flex-end; /* Align radio items vertically if needed */
        flex-direction: row; /* Default, but be explicit */
        justify-content: flex-start; /* Align items to the start (right in RTL) */
    }
     .stRadio label {
         margin-left: 0.5rem; /* Space between radio and label */
         margin-right: 0;
     }


    /* For general RTL text (can be applied via markdown) */
    .rtl-text {
        direction: rtl;
        text-align: right;
    }

    /* For search icon - positioning depends on context */
    .search-icon {
        color: var(--accent-green);
        font-size: 24px;
        /* If placed next to text/input, margin might need adjustment */
        /* margin-right: 0.5rem; -> margin-left: 0.5rem; */
    }

    /* --- Results table (DataFrame) styling --- */
    .dataframe {
        border-radius: 10px !important;
        overflow: hidden;
        direction: rtl; /* Ensure table flows RTL */
    }

    .dataframe th {
        background-color: var(--main-blue);
        color: white;
        text-align: right !important; /* Align header text right */
        /* Add padding if needed for Hebrew text */
        padding: 0.75rem 1rem !important;
    }

    .dataframe td {
        text-align: right !important; /* Align cell text right */
        /* Add padding if needed */
        padding: 0.5rem 1rem !important;
        /* Handle potential line breaks/wrapping */
        white-space: normal; /* Allow wrapping */
        word-wrap: break-word;
    }

    /* --- Force LTR for specific elements like code blocks --- */
    .force-ltr, .stCodeBlock {
        direction: ltr !important;
        text-align: left !important;
    }
    /* Ensure code block content is also LTR */
     .stCodeBlock pre {
         direction: ltr !important;
         text-align: left !important;
     }

     /* Style for generated answer text */
     .answer-text {
         text-align: right; /* Ensure alignment */
         direction: auto; /* Let browser decide based on first strong char */
         white-space: pre-wrap; /* Preserve line breaks */
         line-height: 1.6;
     }

     /* Metric alignment */
     .stMetric {
         direction: rtl; /* Ensure metric container flows RTL */
     }
     .stMetric > label { /* The label above the value */
         text-align: right !important;
         width: 100%; /* Ensure label takes full width for alignment */
     }
     .stMetric > div { /* The container for value and delta */
         justify-content: flex-end; /* Align value/delta to the right */
     }
     .stMetric .metric-container {
          text-align: right; /* Align text within the metric container right */
     }

</style>
""",
    unsafe_allow_html=True,
)
# Create an OpenAI client instance
client = OpenAI(api_key=st.secrets["openai_api_key"])


# Cache the CSV loading and database creation so it runs only once per session
@st.cache_data(show_spinner=False)
def load_csv_to_sqlite(db_path="reports.db"):
    try:
        # Read CSV files with proper encoding
        with st.spinner("📊 Loading data..."):
            enforcement_df = pd.read_csv(
                "אכיפה.csv", encoding="utf-8", low_memory=False
            )
            financial_df = pd.read_csv(
                "תנועות כספיות.csv", encoding="utf-8", low_memory=False
            )
            address_df = pd.read_csv(
                "מאגר כתובות.csv", encoding="utf-8", low_memory=False
            )
            report_df = pd.read_csv("דטא דוחות.csv", encoding="utf-8", low_memory=False)
    except Exception as e:
        st.error(f"Error loading CSV files: {e}")
        return None

    # Create and populate SQLite database
    conn = sqlite3.connect(db_path)
    try:
        enforcement_df.to_sql("enforcement", conn, if_exists="replace", index=False)
        financial_df.to_sql(
            "financial_transactions", conn, if_exists="replace", index=False
        )
        address_df.to_sql("address_database", conn, if_exists="replace", index=False)
        report_df.to_sql("report_data", conn, if_exists="replace", index=False)
        prepare_database_for_date_queries(db_path)
    except Exception as e:
        st.error(f"Error writing to SQLite: {e}")
        conn.close()
        return None

    conn.close()
    return db_path


# Database schema and examples remain the same as your original code
schema = """
Table: enforcement
- "מס' דו''ח": report number
direction: rtl
- תאריך: date
- יום: day
- שעה: time
- "קוד פקח": inspector code
- "שם פקח": inspector name
- "קוד רחוב": street code
- "שם רחוב": street name
- מיקום: location
- "מס' בית": house number
- עבירה: offense
- "מס ' רישוי": vehicle registration number
- סוג: type
- צבע: color
- תוצרת: make
- נכה: disabled
- מבוקש: wanted
- "ננעל/נגרר": locked/towed
- "כרטיס חניה 1": parking card 1
- "כרטיס חניה 2": parking card 2
- "תאריך קובע": determining date
- "לתשלום עד": payment due date
- קנס: fine
- "הערת פקח 1": inspector note 1
- "הערת פקח 2": inspector note 2
- "הערת פקח 3": inspector note 3
- "הערת פקח 4": inspector note 4
- "אזור חניה": parking area
- "אזור פיקוח": supervision area
- "ת''ז": ID number
- "שם משפחה": last name
- "שם פרטי": first name
- רחוב: street
- "'מס": house number
- דירה: apartment
- עיר: city
- מיקוד: postal code
- שולם: paid
- לתשלום: to pay
- "תאריך תשלום": payment date
- "ערעור מתאריך": appeal from date
- "הסבה מתאריך": conversion from date
- "בקשה להישפט": request for trial
- "מספר יחודי": unique number
- "סטטוס לדוח": report status
- פעולה: action
- "מספר שבב": chip number
- "/דוח מוביל": leading report
- "מספר דרכון": passport number

Table: financial_transactions
- "מס' דו''ח": report number
- תאריך: date
- סוג: type
- "ת.תשלום": payment date
- חיוב: charge
- זיכוי: credit
- "ת. פירעון": repayment date

Table: address_database
- "מס' דו''ח": report number
- תאריך: date
- "ת.ז": ID number
- "שם משפחה": last name
- "שם פרטי": first name
- רחוב: street
- מס: house number
- דירה: apartment
- מיקוד: postal code
- "ת.ד": PO box
- עיר: city
- מקור: source

Table: report_data
- "מס' דו''ח": report number
- תאריך: date
- יום: day
- שעה: time
- "קוד פקח": inspector code
- "שם פקח": inspector name
- "קוד רחוב": street code
- "שם רחוב": street name
- מיקום: location
- "מס' בית": house number
- עבירה: offense
- "מס ' רישוי": vehicle registration number
- סוג: type
- צבע: color
- תוצרת: make
- נכה: disabled
- מבוקש: wanted
- "ננעל/נגרר": locked/towed
- "כרטיס חניה 1": parking card 1
- "כרטיס חניה 2": parking card 2
- "תאריך קובע": determining date
- "לתשלום עד": payment due date
- קנס: fine
- "הערת פקח 1": inspector note 1
- "הערת פקח 2": inspector note 2
- "הערת פקח 3": inspector note 3
- "הערת פקח 4": inspector note 4
- "אזור חניה": parking area
- "אזור פיקוח": supervision area
- "ת''ז": ID number
- "שם משפחה": last name
- "שם פרטי": first name
- רחוב: street
- "'מס": house number
- דירה: apartment
- עיר: city
- מיקוד: postal code
- שולם: paid
- לתשלום: to pay
- "תאריך תשלום": payment date
- "ערעור מתאריך": appeal from date
- "הסבה מתאריך": conversion from date
- "בקשה להישפט": request for trial
- "מספר יחודי": unique number
- "הערות לדוח": report notes
- "סטטוס לדוח": report status
- פעולה: action
- "מספר שבב": chip number
- "/דוח מוביל": leading report
- "מספר דרכון": passport number
"""

examples = [
    {
        "question": "What is the fine for report number 123?",
        "sql": "SELECT קנס FROM enforcement WHERE \"מס' דו''ח\" = 123;",
    },
    {
        "question": "How much was paid for report number 456?",
        "sql": "SELECT שולם FROM report_data WHERE \"מס' דו''ח\" = 456;",
    },
    {
        "question": "What is the address for the individual in report number 789?",
        "sql": "SELECT רחוב, \"'מס\", דירה, עיר FROM address_database WHERE \"מס' דו''ח\" = 789;",
    },
    {
        "question": "List all financial transactions for report number 101.",
        "sql": "SELECT * FROM financial_transactions WHERE \"מס' דו''ח\" = 101;",
    },
    {
        "question": "What is the offense and fine for report number 202?",
        "sql": "SELECT עבירה, קנס FROM enforcement WHERE \"מס' דו''ח\" = 202;",
    },
]


def prepare_database_for_date_queries(db_path="reports.db"):
    """
    Adds date index and ensures date columns are properly formatted for SQLite date functions
    """
    conn = sqlite3.connect(db_path)
    try:
        # Create a temporary column with standardized date format if needed
        conn.execute(
            """
        ALTER TABLE enforcement ADD COLUMN date_formatted TEXT;
        """
        )

        # Update the new column with properly formatted dates
        # Assumes your dates are in DD/MM/YYYY format - adjust if different
        conn.execute(
            """
        UPDATE enforcement
        SET date_formatted =
            substr(תאריך, 7, 4) || '-' ||
            substr(תאריך, 4, 2) || '-' ||
            substr(תאריך, 1, 2)
        WHERE תאריך LIKE '__/__/____';
        """
        )

        # Create an index on the new date column for faster queries
        conn.execute(
            """
        CREATE INDEX IF NOT EXISTS idx_enforcement_date ON enforcement (date_formatted);
        """
        )

        # Do the same for report_data table
        conn.execute(
            """
        ALTER TABLE report_data ADD COLUMN date_formatted TEXT;
        """
        )

        conn.execute(
            """
        UPDATE report_data
        SET date_formatted =
            substr(תאריך, 7, 4) || '-' ||
            substr(תאריך, 4, 2) || '-' ||
            substr(תאריך, 1, 2)
        WHERE תאריך LIKE '__/__/____';
        """
        )

        conn.execute(
            """
        CREATE INDEX IF NOT EXISTS idx_report_data_date ON report_data (date_formatted);
        """
        )

        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        # If column already exists, that's fine
        conn.rollback()
    finally:
        conn.close()


def sanitize_sql_query(sql_query):
    """
    Sanitize and fix common SQL query issues
    """
    # Remove any trailing semicolons that could cause 'you can only execute one statement' errors
    sql_query = sql_query.strip()
    if sql_query.endswith(";"):
        sql_query = sql_query[:-1]

    # Remove any additional statements that might be present
    if ";" in sql_query:
        sql_query = sql_query.split(";")[0]

    # Fix common issues with SQLite's handling of division and CAST
    if "CAST(" in sql_query and "AS FLOAT" in sql_query:
        # SQLite doesn't handle CAST(x AS FLOAT) well, replace with CAST(x AS REAL)
        sql_query = sql_query.replace("AS FLOAT", "AS REAL")

    # Handle alternative syntax for percentage calculation
    if "percentage" in sql_query.lower() and "/" in sql_query:
        # Check for potentially problematic division operations
        # SQLite might need explicit casting to avoid integer division
        if not "1.0" in sql_query and not "100.0" in sql_query:
            sql_query = sql_query.replace("* 100", "* 100.0")
            sql_query = sql_query.replace("/ COUNT", "/ CAST(COUNT")
            if not "AS REAL" in sql_query and "))" in sql_query:
                sql_query = sql_query.replace("))", " AS REAL))")

    return sql_query


def execute_sql_query(sql, db_path="reports.db"):
    """
    Execute a SQL query with improved error handling and query sanitization
    """
    # Sanitize the SQL query first
    sanitized_sql = sanitize_sql_query(sql)

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()

        # For debugging, log the sanitized query
        print(f"Executing sanitized query: {sanitized_sql}")

        cursor.execute(sanitized_sql)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        return results, columns
    except sqlite3.Error as e:
        # Enhanced error information for debugging
        error_msg = f"SQLite error: {str(e)}\nQuery: {sanitized_sql}"
        print(error_msg)

        # Try to provide more specific error guidance
        if "no such column" in str(e).lower():
            # Get the column name from the error
            import re

            col_match = re.search(r"no such column: ([^\s]+)", str(e).lower())
            if col_match:
                error_column = col_match.group(1)
                error_msg += f"\n\nThe column '{error_column}' does not exist. Available columns might be different."

        return None, error_msg
    finally:
        conn.close()


# Add a function to handle the specific case of the disabled percentage query
def get_disabled_vehicle_percentage(db_path="reports.db"):
    """
    Specifically handle the calculation of disabled vehicle percentage
    """
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()

        # First, check what values exist in the נכה column
        cursor.execute("SELECT DISTINCT נכה FROM enforcement")
        distinct_values = cursor.fetchall()
        print(f"Distinct values in נכה column: {distinct_values}")

        # Try a simpler query that avoids complex casting
        query = """
        SELECT
            COUNT(*) as total_tickets,
            SUM(CASE WHEN נכה = 'כן' THEN 1
                     WHEN נכה = 'yes' THEN 1
                     WHEN נכה = '1' THEN 1
                     WHEN נכה = 1 THEN 1
                     ELSE 0 END) as disabled_tickets,
            ROUND((SUM(CASE WHEN נכה = 'כן' THEN 1
                         WHEN נכה = 'yes' THEN 1
                         WHEN נכה = '1' THEN 1
                         WHEN נכה = 1 THEN 1
                         ELSE 0 END) * 100.0) / COUNT(*), 2) as percentage
        FROM enforcement
        """

        cursor.execute(query)
        results = cursor.fetchall()
        columns = ["Total Tickets", "Disabled Vehicle Tickets", "Percentage"]

        return results, columns
    except sqlite3.Error as e:
        return None, f"Error calculating disabled percentage: {str(e)}"
    finally:
        conn.close()


def improve_date_examples_in_prompt(schema, examples):
    """
    Update examples to include more robust date-based query examples
    """

    # Add a variety of date examples to help the model understand how to handle dates
    date_examples = [
        {
            "question": "How many tickets were issued in December 2024?",
            "sql": "SELECT COUNT(*) FROM enforcement WHERE strftime('%Y-%m', date_formatted) = '2024-12';",
        },
        {
            "question": "List the inspectors who issued tickets in the last quarter of 2024.",
            "sql": "SELECT DISTINCT \"שם פקח\" FROM enforcement WHERE date_formatted >= '2024-10-01' AND date_formatted <= '2024-12-31';",
        },
        {
            "question": "How many tickets did each inspector issue in the first month of 2024?",
            "sql": "SELECT \"שם פקח\", COUNT(*) as ticket_count FROM enforcement WHERE date_formatted BETWEEN '2024-01-01' AND '2024-01-31' GROUP BY \"שם פקח\" ORDER BY ticket_count DESC;",
        },
    ]

    # Combine existing examples with new date examples
    return examples + date_examples


def generate_better_sql_prompt(question, schema, examples):
    """Enhanced version of generate_sql_prompt with better date handling guidance"""

    # Add information about date columns and specific date handling to the schema
    schema_with_date_info = (
        schema
        + """

Note about Date Handling:
- Dates in the database are in DD/MM/YYYY format but have been converted to YYYY-MM-DD in date_formatted column
- For date queries, use date_formatted column instead of תאריך
- The database contains data from 01/01/2021 to 31/12/2024
- When querying recent data, use specific date literals instead of DATE('now') functions
- Examples of effective date queries:
  * For last month of data: WHERE date_formatted BETWEEN '2024-12-01' AND '2024-12-31'
  * For specific year-month: WHERE strftime('%Y-%m', date_formatted) = '2024-12'
  * For date ranges: WHERE date_formatted BETWEEN '2024-01-01' AND '2024-03-31'
"""
    )

    # Get enhanced examples with better date handling
    enhanced_examples = improve_date_examples_in_prompt(schema, examples)

    # Construct the prompt with schema and examples
    prompt = f"Schema:\n{schema_with_date_info}\n\nExamples:\n"
    for ex in enhanced_examples:
        prompt += f"Question: {ex['question']}\nSQL: {ex['sql']}\n\n"

    # Process the question to check if it's asking about "last month" or recent dates
    is_asking_about_last_month = (
        "חודש האחרון" in question or "last month" in question.lower()
    )

    # Add hint for date handling based on the question
    if is_asking_about_last_month:
        prompt += f"""New Question: {question}

Important: This question is asking about the "last month". Since the database only contains data until 31/12/2024,
interpret "last month" as December 2024 (the most recent month in the database).
Use WHERE date_formatted BETWEEN '2024-12-01' AND '2024-12-31' instead of DATE('now', '-1 month').
"""
    else:
        prompt += f"New Question: {question}\n"

    prompt += "Please generate the SQL query with appropriate date handling as instructed above."
    return prompt


def debug_date_formatting(db_path="reports.db"):
    """
    Function to inspect date formatting issues and provide debugging information
    """
    conn = sqlite3.connect(db_path)
    debug_info = {}

    try:
        cursor = conn.cursor()

        # Check if date_formatted column exists
        cursor.execute("PRAGMA table_info(enforcement)")
        columns = cursor.fetchall()
        has_date_formatted = any(col[1] == "date_formatted" for col in columns)

        debug_info["has_date_formatted_column"] = has_date_formatted

        # Get sample of original dates
        cursor.execute("SELECT תאריך FROM enforcement LIMIT 5")
        original_dates = cursor.fetchall()
        debug_info["sample_original_dates"] = original_dates

        if has_date_formatted:
            # Get sample of formatted dates
            cursor.execute("SELECT תאריך, date_formatted FROM enforcement LIMIT 5")
            formatted_dates = cursor.fetchall()
            debug_info["sample_formatted_dates"] = formatted_dates

            # Check for nulls in date_formatted
            cursor.execute(
                "SELECT COUNT(*) FROM enforcement WHERE date_formatted IS NULL"
            )
            null_count = cursor.fetchone()[0]
            debug_info["null_formatted_dates"] = null_count

            # Count records from December 2024
            cursor.execute(
                "SELECT COUNT(*) FROM enforcement WHERE date_formatted BETWEEN '2024-12-01' AND '2024-12-31'"
            )
            dec_2024_count = cursor.fetchone()[0]
            debug_info["december_2024_records"] = dec_2024_count

    except sqlite3.Error as e:
        debug_info["error"] = str(e)
    finally:
        conn.close()

    return debug_info


def generate_textual_answer(question, sql_query, results, columns, language="hebrew"):
    """
    Generate a natural language answer based on the SQL query results and stream the answer.
    """
    # Create a DataFrame for easier processing
    df = pd.DataFrame(results, columns=columns)

    # Generate a description of the results
    if df.empty:
        result_description = "No data was found for this query."
    else:
        # Basic description of what's in the dataframe
        num_rows = len(df)
        result_description = f"Found {num_rows} rows of data.\n\n"

        # For small datasets, include the full data in a readable format
        if num_rows <= 20:
            result_description += df.to_string(index=False)
        else:
            # For larger datasets, summarize the data
            result_description += f"Sample of data (first 5 rows):\n{df.head(5).to_string(index=False)}\n\n"

            # Add some basic statistics if appropriate
            numeric_columns = df.select_dtypes(include=["number"]).columns
            if len(numeric_columns) > 0:
                result_description += "Summary statistics for numeric columns:\n"
                result_description += df[numeric_columns].describe().to_string()

    # Prepare the prompt for OpenAI to generate a conversational answer
    prompt = f"""
The user asked the following question:
{question}

The following SQL query was executed to answer this question:
{sql_query}

The query returned the following results:
{result_description}

Please generate a natural, conversational answer that explains these results in a way that directly answers the original question.
The answer should be in {language} and should be easy to understand for someone who doesn't know SQL.
"""
    answer = ""
    placeholder = st.empty()  # Placeholder for streaming text

    try:
        # Call the OpenAI API with stream=True to get a streaming response
        with st.spinner("AI is generating your answer..."):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that explains database query results.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                stream=True,  # Enable streaming mode
            )

            # Add a pulsing dot animation while waiting
            answer_container = st.empty()
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta is None:
                    continue
                # If this chunk contains regular content:
                if delta.content is not None:
                    token = delta.content
                    answer += token
                    answer_container.markdown(
                        f"<div dir='auto' class='answer-text'>{answer}</div>",
                        unsafe_allow_html=True,
                    )

        return answer
    except Exception as e:
        error_text = f"Error generating textual answer: {e}"
        placeholder.markdown(error_text)
        return error_text


def add_debugging_tools(db_path):
    """Add debugging tools to the Streamlit app"""

    with st.expander("⚙️ Database Diagnostics", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Check Date Formatting", type="primary"):
                debug_info = debug_date_formatting(db_path)
                st.json(debug_info)

        with col2:
            if st.button("Run Sample Date Query (Dec 2024)", type="primary"):
                conn = sqlite3.connect(db_path)
                try:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT \"שם פקח\", COUNT(*) as count FROM enforcement WHERE date_formatted BETWEEN '2024-12-01' AND '2024-12-31' GROUP BY \"שם פקח\" ORDER BY count DESC"
                    )
                    results = cursor.fetchall()
                    if results:
                        df = pd.DataFrame(
                            results, columns=["Inspector", "Ticket Count"]
                        )
                        st.dataframe(df)
                    else:
                        st.info("No data found for December 2024")
                except Exception as e:
                    st.error(f"Query error: {e}")
                finally:
                    conn.close()


def create_feature_cards():
    """Create cards for quick access to key features"""
    st.markdown(
        "<h3 style='margin-top: 0; margin-bottom: 1rem;'>Quick Actions</h3>",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
        <div class="card" style="text-align: center; cursor: pointer;">
            <div style="font-size: 2rem; color: #0F75BC; margin-bottom: 0.5rem;">🔍</div>
            <h4 style="margin: 0.5rem 0;">Search Reports</h4>
            <p style="font-size: 0.8rem; color: #666;">Find reports by number or details</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="card" style="text-align: center; cursor: pointer;">
            <div style="font-size: 2rem; color: #0F75BC; margin-bottom: 0.5rem;">📊</div>
            <h4 style="margin: 0.5rem 0;">View Statistics</h4>
            <p style="font-size: 0.8rem; color: #666;">See enforcement trends and data</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="card" style="text-align: center; cursor: pointer;">
            <div style="font-size: 2rem; color: #0F75BC; margin-bottom: 0.5rem;">📝</div>
            <h4 style="margin: 0.5rem 0;">Generate Reports</h4>
            <p style="font-size: 0.8rem; color: #666;">Create custom reports and analysis</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
        <div class="card" style="text-align: center; cursor: pointer;">
            <div style="font-size: 2rem; color: #0F75BC; margin-bottom: 0.5rem;">⚙️</div>
            <h4 style="margin: 0.5rem 0;">Settings</h4>
            <p style="font-size: 0.8rem; color: #666;">Configure application preferences</p>
        </div>
        """,
            unsafe_allow_html=True,
        )


def main():
    # Custom header with logo
    st.markdown(
        """
    <div class="main-header">
        <div style="display: flex; align-items: center;">
            <h1>MGroup Reports AI</h1>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Language toggle
    # col1, col2 = st.columns([4, 1])
    # with col2:
    #     language = st.radio(
    #         "Language / שפה",
    #         ["עברית", "English"],
    #         horizontal=True,
    #         label_visibility="collapsed",
    #     )

    # Create feature cards for quick access
    # create_feature_cards()

    # Create tabs for different functionality
    tabs = st.tabs(["🔍 שאילתות חכמות", "📊 ניתוח נתונים", "📋 דוחות", "⚙️ הגדרות"])

    with tabs[0]:  # AI Query tab
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        # Load the CSV files into SQLite and check for errors
        db_path = load_csv_to_sqlite()
        if db_path is None:
            st.error("❌ Failed to load the CSV files into the SQLite database.")
            return

        # Search box with icon
        st.markdown("<h3>שאל שאלה על נתוני הדוחות</h3>", unsafe_allow_html=True)
        question = st.text_input(
            "הכנס שאלה בעברית או באנגלית...",
            placeholder="לדוגמה: כמה דוחות ניתנו בחודש דצמבר 2024?",
            label_visibility="collapsed",
        )

        # Example questions as chips
        st.markdown(
            "<p style='margin-top: 0.5rem; margin-bottom: 1rem;'><b>שאלות לדוגמה:</b></p>",
            unsafe_allow_html=True,
        )
        example_cols = st.columns(3)
        with example_cols[0]:
            st.markdown(
                """
            <div style="background-color: #f0f7fa; padding: 8px 15px; border-radius: 20px; margin-bottom: 10px; cursor: pointer; text-align: center;">
                כמה דוחות ניתנו בדצמבר 2024?
            </div>
            """,
                unsafe_allow_html=True,
            )
        with example_cols[1]:
            st.markdown(
                """
            <div style="background-color: #f0f7fa; padding: 8px 15px; border-radius: 20px; margin-bottom: 10px; cursor: pointer; text-align: center;">
                מהם סכומי הקנסות לפי פקח?
            </div>
            """,
                unsafe_allow_html=True,
            )
        with example_cols[2]:
            st.markdown(
                """
            <div style="background-color: #f0f7fa; padding: 8px 15px; border-radius: 20px; margin-bottom: 10px; cursor: pointer; text-align: center;">
                כמה רכבי נכים קיבלו דוחות?
            </div>
            """,
                unsafe_allow_html=True,
            )

        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            search_button = st.button(
                "🔍 חפש", type="primary", use_container_width=True
            )
        with col2:
            clear_button = st.button("🔄 נקה", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Process the query if button is clicked
        if search_button and question.strip():
            st.markdown("<div class='card'>", unsafe_allow_html=True)

            # Animated loading indicator
            with st.spinner("⏳ מעבד את השאלה שלך..."):
                # Check for specific questions that need special handling
                is_disabled_percentage_question = any(
                    term in question.lower()
                    for term in ["נכים", "disabled", "percentage", "אחוז"]
                )

                if is_disabled_percentage_question and "רכב" in question:
                    # Direct handling for the disabled vehicles percentage question
                    results, columns_or_error = get_disabled_vehicle_percentage(db_path)

                    if results is not None:
                        df = pd.DataFrame(results, columns=columns_or_error)

                        # Create a nice visualization card for the results
                        st.markdown("### 📊 תוצאות הניתוח")

                        # Create metrics display
                        metric_cols = st.columns(3)
                        with metric_cols[0]:
                            st.metric(
                                label="סה״כ דוחות", value=f"{int(df.iloc[0][0]):,}"
                            )
                        with metric_cols[1]:
                            st.metric(
                                label="דוחות לרכבי נכים",
                                value=f"{int(df.iloc[0][1]):,}",
                            )
                        with metric_cols[2]:
                            st.metric(label="אחוז מסך הכל", value=f"{df.iloc[0][2]}%")

                        # Create expandable raw data section
                        with st.expander("הצג נתונים גולמיים", expanded=False):
                            st.dataframe(df)

                        # Generate a textual answer
                        language = (
                            "hebrew"
                            if any("\u0590" <= c <= "\u05FF" for c in question)
                            else "english"
                        )
                        sql_query = (
                            "Special query for calculating disabled vehicle percentage"
                        )

                        st.markdown("### 📝 תשובה")
                        textual_answer = generate_textual_answer(
                            question, sql_query, results, columns_or_error, language
                        )
                    else:
                        st.error(f"Error: {columns_or_error}")
                else:
                    # Normal flow for other questions
                    # Generate the prompt for the AI
                    prompt = generate_better_sql_prompt(question, schema, examples)

                    with st.expander("פרטי SQL", expanded=False):
                        st.write("**Prompt for SQL Generation:**")
                        st.code(prompt, language="text")

                    try:
                        # Call the OpenAI API to generate the SQL query
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "You are a helpful assistant that generates SQL queries for SQLite based on natural language "
                                        "questions and a given database schema. Use specific date literals rather than SQLite date functions."
                                    ),
                                },
                                {"role": "user", "content": prompt},
                            ],
                            max_tokens=150,
                            temperature=0.0,
                        )
                        full_response = response.choices[0].message.content.strip()
                    except Exception as e:
                        st.error(f"Error communicating with OpenAI: {e}")
                        return

                    # Extract the SQL query using regex; allow for trailing semicolons
                    sql_match = re.search(
                        r"(SELECT.*?;?)$", full_response, re.DOTALL | re.IGNORECASE
                    )
                    if sql_match:
                        sql_query = sql_match.group(1).strip()
                    else:
                        sql_query = full_response  # Fallback if regex fails

                    with st.expander("קוד SQL שנוצר", expanded=False):
                        st.code(sql_query, language="sql")

                    # Execute the SQL query and display the results
                    results, columns_or_error = execute_sql_query(sql_query, db_path)
                    if results is not None:
                        if results:
                            df = pd.DataFrame(results, columns=columns_or_error)

                            # Nice results display
                            st.markdown("### 📊 תוצאות הניתוח")

                            # For small result sets, show as a styled table
                            if len(df) <= 10:
                                st.dataframe(
                                    df,
                                    use_container_width=True,
                                    hide_index=True,
                                    column_config={
                                        col: st.column_config.Column(
                                            col, help=f"Column: {col}"
                                        )
                                        for col in df.columns
                                    },
                                )
                            else:
                                # For larger datasets, show summary and expandable full results
                                st.write(f"**נמצאו {len(df)} שורות**")
                                tab1, tab2 = st.tabs(["תצוגת טבלה", "סטטיסטיקה"])

                                with tab1:
                                    st.dataframe(
                                        df.head(10),
                                        use_container_width=True,
                                        hide_index=True,
                                    )
                                    st.caption("מוצגות 10 השורות הראשונות בלבד")

                                with tab2:
                                    # Show summary statistics if there are numeric columns
                                    numeric_cols = df.select_dtypes(
                                        include=["number"]
                                    ).columns
                                    if len(numeric_cols) > 0:
                                        st.write("**סטטיסטיקה:**")
                                        st.dataframe(df[numeric_cols].describe())

                                with st.expander("הצג את כל הנתונים", expanded=False):
                                    st.dataframe(
                                        df, use_container_width=True, hide_index=True
                                    )

                            # Generate a textual answer
                            language = (
                                "hebrew"
                                if any("\u0590" <= c <= "\u05FF" for c in question)
                                else "english"
                            )

                            st.markdown("### 📝 תשובה")
                            textual_answer = generate_textual_answer(
                                question, sql_query, results, columns_or_error, language
                            )
                        else:
                            st.info("לא נמצאו נתונים לשאלה שהוזנה.")
                    else:
                        st.error(f"שגיאה בביצוע שאילתת SQL: {columns_or_error}")

            st.markdown("</div>", unsafe_allow_html=True)

    with tabs[1]:  # Data Analysis tab
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("## 📊 ניתוח נתונים")
        st.write("בחר סוג ניתוח מהאפשרויות הבאות:")

        analysis_type = st.selectbox(
            "סוג ניתוח",
            [
                "התפלגות דוחות לפי חודשים",
                "השוואת פקחים",
                "מגמות לפי אזורים",
                "ניתוח תשלומים",
            ],
        )

        st.write("תצוגת ניתוח תופיע כאן לאחר פיתוח הפונקציונליות.")
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[2]:  # Reports tab
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("## 📋 דוחות")

        report_cols = st.columns(2)
        with report_cols[0]:
            report_type = st.selectbox(
                "סוג דו״ח",
                ["דו״ח חודשי", "דו״ח רבעוני", "דו״ח שנתי", "דו״ח מותאם אישית"],
            )

        with report_cols[1]:
            period = st.selectbox(
                "תקופה",
                [
                    "ינואר 2024",
                    "פברואר 2024",
                    "מרץ 2024",
                    "אפריל 2024",
                    "מאי 2024",
                    "יוני 2024",
                    "יולי 2024",
                    "אוגוסט 2024",
                    "ספטמבר 2024",
                    "אוקטובר 2024",
                    "נובמבר 2024",
                    "דצמבר 2024",
                ],
            )

        generate_report = st.button("📄 הפק דו״ח", type="primary")

        if generate_report:
            st.info("פונקציונליות הפקת דוחות תתווסף בגרסה הבאה.")

        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[3]:  # Settings tab
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("## ⚙️ הגדרות")

        settings_tabs = st.tabs(["הגדרות כלליות", "מודל AI", "יצוא/יבוא", "עזרה"])

        with settings_tabs[0]:
            st.write("הגדרות כלליות של המערכת:")
            st.toggle("הפעל מצב כהה", value=False)
            st.toggle("הצג הערות פקח בתוצאות", value=True)
            st.toggle("שמור היסטוריית שאילתות", value=True)

        with settings_tabs[1]:
            st.write("הגדרות מודל AI:")
            st.selectbox("מודל OpenAI", ["GPT-4o", "GPT-3.5 Turbo"])
            st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
            st.text_input("OpenAI API Key (השאר ריק כדי להשתמש בברירת המחדל)")

        with settings_tabs[2]:
            st.write("אפשרויות יצוא/יבוא:")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "יצא נתונים", data="sample data", file_name="export.csv"
                )
            with col2:
                st.file_uploader("העלה קובץ CSV", type=["csv"])

        with settings_tabs[3]:
            st.write("עזרה ותמיכה:")
            st.markdown(
                """
            לתמיכה טכנית:
            - שלח מייל ל: support@example.com
            - התקשר: 08-123-4567

            [מדריך למשתמש](https://example.com/guide)
            """
            )

        # Add debugging tools in a collapsible section
        add_debugging_tools(db_path)
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
