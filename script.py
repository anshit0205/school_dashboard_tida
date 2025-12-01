import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
from streamlit_autorefresh import st_autorefresh   

# ⏱ Auto-refresh interval (seconds)
AUTO_REFRESH_SECONDS = 20

# 📄 Google Sheets – published CSV link
SHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vRmyabKeDmpd1u9oWNyiZyIE1dUi4v-2yxbE0m1R_rA11AvKUAKTHm-PmdzQjEIHj1YXjKpoKbFk1O6/pub?output=csv"
)

# Page configuration – MUST be first Streamlit command
st.set_page_config(page_title="Activity Enrollment Analytics", layout="wide", page_icon="📊")

# Auto-refresh the whole app every X seconds
st_autorefresh(interval=AUTO_REFRESH_SECONDS * 1000, key="auto_refresh")

# Session state for change detection
if "previous_df_signature" not in st.session_state:
    st.session_state.previous_df_signature = None

# Custom CSS
st.markdown("""
    <style>
    .big-metric {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: -10px;
    }
    .kpi-card {
        background: linear-gradient(135deg, #1f2933 0%, #3b82f6 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .section-header {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 20px 0;
        border: 1px solid #e5e7eb;
    }
    .section-header h2 {
        margin: 0;
        font-size: 1.3rem;
        color: #111827;
    }
    .multi-activity-badge {
        background-color: #fef3c7;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.85em;
        color: #92400e;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
    /* Make the tab bar look cleaner */
    .stTabs [role="tablist"] {
        gap: 0.5rem;
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 0.25rem;
    }

    .stTabs [role="tab"] {
        padding: 0.4rem 0.9rem;
        border-radius: 999px;
        background-color: #f3f4f6;
        color: #4b5563;
        font-weight: 500;
        border: 1px solid transparent;
    }

    .stTabs [role="tab"]:hover {
        background-color: #e5e7eb;
        color: #111827;
    }

    .stTabs [role="tab"][aria-selected="true"] {
        background: #fee2e2;
        color: #b91c1c;
        border-color: #fca5a5;
    }
    </style>
""", unsafe_allow_html=True)

# ============= HELPER FUNCTIONS =============
def extract_package_duration(remarks):
    """Extract package duration from remarks with improved pattern matching."""
    if pd.isna(remarks):
        return 1
    remarks = str(remarks).lower()
    
    # Check for yearly/annual first
    if 'yearly' in remarks or 'annual' in remarks or '12 month' in remarks or 'twelve month' in remarks:
        return 12
    
    # Check for specific month patterns
    month_patterns = [
        (r'(\d+)\s*month', lambda x: int(x)),  # "6 month", "3 month"
        (r'(six|6)\s*month', lambda x: 6),
        (r'(five|5)\s*month', lambda x: 5),
        (r'(four|4)\s*month', lambda x: 4),
        (r'(three|3)\s*month', lambda x: 3),
        (r'(two|2)\s*month', lambda x: 2),
    ]
    
    for pattern, converter in month_patterns:
        match = re.search(pattern, remarks)
        if match:
            try:
                if pattern == r'(\d+)\s*month':
                    return int(match.group(1))
                else:
                    return converter(match.group(1))
            except:
                continue
    
    return 1

def parse_activities(activity_string):
    """Parse activity string and return list of individual activities."""
    if pd.isna(activity_string) or activity_string == '':
        return []
    
    # Split by common separators
    activities = re.split(r'[+,&/|]', str(activity_string))
    # Clean and filter
    activities = [a.strip() for a in activities if a.strip()]
    return activities

def is_multi_activity(activity_string):
    """Check if student has opted for multiple activities."""
    activities = parse_activities(activity_string)
    return len(activities) > 1

def calculate_renewal_date(payment_date, fees_due_date, package_duration):
    """
    Calculate renewal date based on fees due date + package duration.
    If fees_due_date is 26 Jan and package is 2 months, renewal is 26 March.
    """
    if pd.isna(fees_due_date) or pd.isna(package_duration):
        return pd.NaT
    
    try:
        renewal = fees_due_date + pd.DateOffset(months=int(package_duration))
        return renewal
    except:
        return pd.NaT

def calculate_retention_risk(days):
    """Calculate retention risk score."""
    if pd.isna(days):
        return 'Unknown'
    if days < 0:
        return 'Critical'  # Overdue - needs immediate action
    elif days <= 14:
        return 'High'      # Due within 2 weeks - proactive outreach needed
    else:
        return 'Active'    # More than 2 weeks away
def clean_phone_number(num):
    """Keep only digits, normalise Indian mobile numbers to 91xxxxxxxxxx."""
    if pd.isna(num) or str(num).strip() == "":
        return ""
    s = re.sub(r"\D", "", str(num))  # remove non-digits
    # If it's a 10-digit mobile, prefix country code
    if len(s) == 10:
        s = "91" + s
    return s
    

# ============= DATA LOADING =============
@st.cache_data
def load_data(cache_bust: int):
    try:
        url = SHEET_CSV_URL + f"&cacheBust={cache_bust}"
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()

        # Treat "Date" as Payment Date
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

        # Extract package duration from Remarks
        if 'Remarks' in df.columns:
            df['Package Duration (Months)'] = df['Remarks'].apply(extract_package_duration)
        else:
            df['Package Duration (Months)'] = 1

        # Handle Fees Due Date
        if 'Fees Due Date' in df.columns:
            df['Fees Due Date'] = pd.to_datetime(df['Fees Due Date'], dayfirst=True, errors='coerce')
        else:
            df['Fees Due Date'] = pd.NaT

        # Auto-calc Fees Due Date where missing
        if 'Date' in df.columns:
            def infer_due(row):
                if pd.notna(row['Fees Due Date']):
                    return row['Fees Due Date']
                if pd.notna(row['Date']) and pd.notna(row['Package Duration (Months)']):
                    return row['Date'] + pd.DateOffset(months=int(row['Package Duration (Months)']))
                return pd.NaT
            df['Fees Due Date'] = df.apply(infer_due, axis=1)

        # Calculate Renewal Date (Fees Due Date + Package Duration)
        df['Renewal Date'] = df.apply(
            lambda row: calculate_renewal_date(
                row['Date'], 
                row['Fees Due Date'], 
                row['Package Duration (Months)']
            ), 
            axis=1
        )

        # Multi-Activity Analysis
        if 'Activity Opted' in df.columns:
            df['Activity List'] = df['Activity Opted'].apply(parse_activities)
            df['Activity Count'] = df['Activity List'].apply(len)
            df['Is Multi-Activity'] = df['Activity Opted'].apply(is_multi_activity)
        else:
            df['Activity List'] = None
            df['Activity Count'] = 0
            df['Is Multi-Activity'] = False

        # Financial metrics
        if 'Amount' in df.columns:
            # For multi-activity, divide amount by number of activities for per-activity revenue
            df['Amount Per Activity'] = df.apply(
                lambda row: row['Amount'] / row['Activity Count'] if row['Activity Count'] > 0 else row['Amount'],
                axis=1
            )
            df['Monthly Equivalent Fee'] = (df['Amount'] / df['Package Duration (Months)']).round(2)
            df['Monthly Fee Per Activity'] = (df['Amount Per Activity'] / df['Package Duration (Months)']).round(2)
        else:
            df['Amount Per Activity'] = 0.0
            df['Monthly Equivalent Fee'] = 0.0
            df['Monthly Fee Per Activity'] = 0.0

        # Use max date from dataset as "today" for consistency
        today = df['Date'].max().normalize() if 'Date' in df.columns and not df['Date'].isna().all() else pd.Timestamp.now().normalize()

        # Days Until Renewal (based on Renewal Date, not Fees Due Date)
        if 'Renewal Date' in df.columns:
            df['Days Until Renewal'] = (df['Renewal Date'] - today).dt.days
            df['Retention Risk'] = df['Days Until Renewal'].apply(calculate_retention_risk)
        else:
            df['Days Until Renewal'] = pd.NA
            df['Retention Risk'] = 'Unknown'

        # Days Until Fees Due
        if 'Fees Due Date' in df.columns:
            df['Days Until Fees Due'] = (df['Fees Due Date'] - today).dt.days

        # Days Since Payment (Customer Tenure)
        if 'Date' in df.columns:
            df['Days Since Payment'] = (today - df['Date']).dt.days
        else:
            df['Days Since Payment'] = pd.NA

        # Payment Age cohorts
        df['Payment Age Bucket'] = pd.cut(
            df['Days Since Payment'],
            bins=[-1, 30, 90, 180, 365*20],
            labels=['0-30 days', '31-90 days', '91-180 days', '180+ days']
        )

        # Payment Status
        if 'Amount' in df.columns:
            df['Payment Status'] = df['Amount'].apply(
                lambda x: 'Unpaid / Zero' if (pd.isna(x) or x <= 0) else 'Paid'
            )
        else:
            df['Payment Status'] = 'Unknown'

        # Calculate months overdue for critical customers
        def calculate_months_overdue(row):
            if row.get('Retention Risk') == 'Critical' and row.get('Payment Status') == 'Paid':
                if pd.notna(row.get('Renewal Date')):
                    months_overdue = max(1, int((today - row['Renewal Date']).days / 30))
                    return months_overdue
            return 0
        
        df['Months Overdue'] = df.apply(calculate_months_overdue, axis=1)
        
        # Due Amount (multiply monthly fee by months overdue)
        df['Due Amount'] = df.apply(
            lambda row: row['Monthly Equivalent Fee'] * row['Months Overdue']
            if row['Months Overdue'] > 0
            else 0,
            axis=1
        )

        # Fill NaN values for text columns
        text_cols = ['Class', 'Activity Opted', 'Mode of Payment', 'Remarks',
                     'School', 'Name of Parent', 'Name of Child',"Father's Number"]
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna('')
        if "Father's Number" in df.columns:
            df["Father's Number"] = df["Father's Number"].apply(clean_phone_number)
        return df
    
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

# ---------- CALL load_data HERE ----------
# This number changes every AUTO_REFRESH_SECONDS seconds
cache_bust_bucket = int(datetime.now().timestamp() // AUTO_REFRESH_SECONDS)
df = load_data(cache_bust_bucket)

# ---------- CHANGE DETECTION FOR "New data loaded ✔" ----------
data_changed = False

# Drop list-type columns (like Activity List) before hashing to avoid "unhashable type: list"
sig_df = df.copy()
if 'Activity List' in sig_df.columns:
    sig_df = sig_df.drop(columns=['Activity List'])

# Build a stable signature of the dataframe
current_signature = hash(pd.util.hash_pandas_object(sig_df, index=False).sum())

if st.session_state.previous_df_signature is None:
    st.session_state.previous_df_signature = current_signature
else:
    if st.session_state.previous_df_signature != current_signature:
        data_changed = True
        st.session_state.previous_df_signature = current_signature

if data_changed:
    st.success("📡 New data loaded ✔")

st.caption(f"Last data pull from Google Sheet: {datetime.now().strftime('%d-%b-%Y %H:%M:%S')}")

# Create expanded dataframe for multi-activity analysis
def create_activity_expanded_df(df):
    """Create a row for each individual activity when students have multiple activities."""
    expanded_rows = []
    
    for idx, row in df.iterrows():
        activities = row['Activity List']
        if activities and len(activities) > 0:
            for activity in activities:
                new_row = row.copy()
                new_row['Individual Activity'] = activity
                new_row['Original Activity String'] = row['Activity Opted']
                expanded_rows.append(new_row)
        else:
            new_row = row.copy()
            new_row['Individual Activity'] = row['Activity Opted']
            new_row['Original Activity String'] = row['Activity Opted']
            expanded_rows.append(new_row)
    
    return pd.DataFrame(expanded_rows)

df_expanded = create_activity_expanded_df(df)

# Verify columns
required_cols = ['Name of Child', 'Activity Opted', 'Amount', 'Date', 'Mode of Payment']
if not all(col in df.columns for col in required_cols):
    st.error("❌ Missing required columns")
    st.stop()

has_school = 'School' in df.columns and not df['School'].eq('').all()
has_class = 'Class' in df.columns and not df['Class'].eq('').all()

# ============= HEADER =============
st.title("📊 Activity Enrollment Analytics Dashboard")
st.markdown("#### Payments, Renewals & Retention Overview")
st.caption("Note: **Date** is treated as the *Payment Date*. 'Today' is set to the latest payment date in the dataset.")
st.markdown("---")

# ============= SIDEBAR =============
st.sidebar.image("https://img.icons8.com/fluency/96/000000/filter.png", width=50)
st.sidebar.title("Filters")

# Date filter (by Payment Date)
date_filter = st.sidebar.selectbox(
    "Time Period (by Payment Date)",
    ["All Time", "Last 7 Days", "Last 14 Days", "Last 30 Days", "Last 90 Days", "Custom"]
)

if date_filter == "Custom":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start = st.date_input("From", value=df['Date'].min())
    with col2:
        end = st.date_input("To", value=df['Date'].max())
    filtered_df = df[(df['Date'] >= pd.to_datetime(start)) & (df['Date'] <= pd.to_datetime(end))]
else:
    days_map = {"Last 7 Days": 7, "Last 14 Days": 14, "Last 30 Days": 30, "Last 90 Days": 90}
    if date_filter in days_map:
        cutoff = df['Date'].max() - timedelta(days=days_map[date_filter])
        filtered_df = df[df['Date'] >= cutoff]
    else:
        filtered_df = df.copy()

st.sidebar.markdown("---")

# Activity filter (individual activities)
all_individual_activities = sorted(list(set([act for acts in df['Activity List'] for act in acts if act])))
if all_individual_activities:
    selected_activities = st.sidebar.multiselect(
        "Activities (Individual)", 
        ["All"] + all_individual_activities, 
        default=["All"]
    )
    if "All" not in selected_activities:
        # Filter students who have any of the selected activities
        filtered_df = filtered_df[filtered_df['Activity List'].apply(
            lambda acts: any(act in selected_activities for act in acts) if acts else False
        )]

# Multi-Activity Filter
st.sidebar.markdown("---")
multi_activity_filter = st.sidebar.radio(
    "Activity Type",
    ["All", "Single Activity Only", "Multi-Activity Only"]
)
if multi_activity_filter == "Single Activity Only":
    filtered_df = filtered_df[~filtered_df['Is Multi-Activity']]
elif multi_activity_filter == "Multi-Activity Only":
    filtered_df = filtered_df[filtered_df['Is Multi-Activity']]

# School filter
if 'School' in df.columns:
    schools = sorted([s for s in df['School'].unique() if s and str(s).strip() != ''])
    if schools:
        selected_schools = st.sidebar.multiselect(
            "Schools", 
            ["All"] + schools, 
            default=["All"]
        )
        if "All" not in selected_schools:
            filtered_df = filtered_df[filtered_df['School'].isin(selected_schools)]

# Class filter
if has_class:
    classes = sorted([c for c in df['Class'].unique() if c])
    if classes:
        selected_classes = st.sidebar.multiselect(
            "Classes", 
            ["All"] + classes, 
            default=["All"]
        )
        if "All" not in selected_classes:
            filtered_df = filtered_df[filtered_df['Class'].isin(selected_classes)]

# Mode of Payment filter
payment_modes = sorted([p for p in df['Mode of Payment'].unique() if p])
if payment_modes:
    selected_payments = st.sidebar.multiselect(
        "Payment Methods", 
        ["All"] + payment_modes, 
        default=["All"]
    )
    if "All" not in selected_payments:
        filtered_df = filtered_df[filtered_df['Mode of Payment'].isin(selected_payments)]

# Retention Risk filter
if 'Retention Risk' in df.columns:
    st.sidebar.markdown("---")
    risk_levels = st.sidebar.multiselect(
        "Retention Status",
        ["All", "Active", "High", "Critical", "Unknown"],
        default=["All"]
    )
    if "All" not in risk_levels:
        filtered_df = filtered_df[filtered_df['Retention Risk'].isin(risk_levels)]

# Payment Status filter
if 'Payment Status' in df.columns:
    st.sidebar.markdown("---")
    pay_status = st.sidebar.multiselect(
        "Payment Status",
        ["All", "Paid", "Unpaid / Zero"],
        default=["All"]
    )
    if "All" not in pay_status:
        filtered_df = filtered_df[filtered_df['Payment Status'].isin(pay_status)]

# Admission Type filter
if 'Type' in df.columns:
    st.sidebar.markdown("---")
    admission_types = sorted([t for t in df['Type'].unique() if t and str(t).strip() != ''])
    if admission_types:
        selected_types = st.sidebar.multiselect(
            "Admission Type",
            ["All"] + admission_types,
            default=["All"]
        )
        if "All" not in selected_types:
            filtered_df = filtered_df[filtered_df['Type'].isin(selected_types)]

st.sidebar.markdown("---")
with st.sidebar.expander("📘 Dashboard Guide"):
    st.markdown("""
    **Key Concepts:**
    - **MRR**: Monthly Recurring Revenue
    - **Renewal Date**: Fees Due Date + Package Duration
    - **Multi-Activity**: Students enrolled in 2+ activities
    - **Retention Risk**: Based on days until renewal
    - **Months Overdue**: How many months past renewal date
    
    **Risk Levels:**
    - 🔴 Critical: Overdue (negative days)
    - 🟡 High: Due within 14 days
    - 🟢 Active: More than 14 days away
    """)

with st.sidebar.expander("🔄 Monthly Maintenance Guide"):
    st.markdown("""
    **When Customer Pays Renewal:**
    
    1. **Add NEW row** in Google Sheet with:
       - Same Name, School, Class, Activity
       - New **Date** (payment date)
       - New **Amount** paid
       - New **Package Duration** in Remarks
       - New **Fees Due Date**
    
    2. **Keep old row** for history tracking
    
    **Track New Admissions:**
    - Add column "Type" = "New" or "Renewal"
    - Or use first occurrence of student name
    
    **Outstanding Dues Calculation:**
    - System now counts ALL months overdue
    - Example: 4 months late = 4 × Monthly Fee
    
    **Best Practice:**
    - Weekly: Check Critical renewals
    - Monthly: Export and archive data
    - Each payment: Add new row (don't overwrite)
    """)


# Create filtered expanded df
filtered_df_expanded = create_activity_expanded_df(filtered_df)

# ============= KEY METRICS SECTION =============
st.markdown('<div class="section-header"><h2>Key Performance Indicators (KPIs)</h2></div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_students = len(filtered_df)
    multi_activity_students = len(filtered_df[filtered_df['Is Multi-Activity']])
    multi_pct = (multi_activity_students / total_students * 100) if total_students > 0 else 0
    st.metric(
        "Total Students",
        f"{total_students}",
        f"{multi_activity_students} multi-activity ({multi_pct:.1f}%)"
    )

with col2:
    total_activity_enrollments = filtered_df['Activity Count'].sum()
    st.metric(
        "Total Activity Enrollments",
        f"{total_activity_enrollments:.0f}",
        f"Avg: {total_activity_enrollments/total_students:.2f} per student" if total_students > 0 else "N/A"
    )

with col3:
    total_revenue = filtered_df['Amount'].sum()
    st.metric(
        "Total Revenue Collected",
        f"₹{total_revenue:,.0f}"
    )

with col4:
    monthly_rev = filtered_df['Monthly Equivalent Fee'].sum()
    st.metric(
        "MRR",
        f"₹{monthly_rev:,.0f}",
        help="Monthly Recurring Revenue"
    )

with col5:
    if 'Due Amount' in filtered_df.columns:
        overdue = filtered_df['Due Amount'].sum()
        overdue_pct = (overdue / total_revenue * 100) if total_revenue > 0 else 0
        st.metric(
            "Outstanding Dues",
            f"₹{overdue:,.0f}",
            f"{overdue_pct:.1f}% of collected",
            delta_color="inverse"
        )
    else:
        st.metric("Outstanding Dues", "₹0")

st.markdown("---")

# Alert for high overdue amounts
if 'Due Amount' in filtered_df.columns:
    total_overdue = filtered_df['Due Amount'].sum()
    overdue_students = len(filtered_df[filtered_df['Due Amount'] > 0])
    if total_overdue > 0:
        avg_months_overdue = filtered_df[filtered_df['Months Overdue'] > 0]['Months Overdue'].mean()
        if overdue_students > 0:
            st.error(f"""
            ⚠️ **COLLECTION ALERT**: {overdue_students} students have ₹{total_overdue:,.0f} in outstanding dues 
            (Average: {avg_months_overdue:.1f} months overdue). Review Priority Follow-up List below immediately!
            """)

# Secondary KPIs
col6, col7, col8, col9, col10 = st.columns(5)

with col6:
    if 'Retention Risk' in filtered_df.columns:
        critical = len(filtered_df[filtered_df['Retention Risk'] == 'Critical'])
        high = len(filtered_df[filtered_df['Retention Risk'] == 'High'])
        churn_risk = critical + high
        churn_rate = (churn_risk / total_students * 100) if total_students > 0 else 0
        st.metric(
            "Students Needing Follow-up",
            f"{churn_risk}",
            f"{churn_rate:.1f}% of base"
        )

with col7:
    multi_month = len(filtered_df[filtered_df['Package Duration (Months)'] > 1])
    multi_pct = (multi_month / total_students * 100) if total_students > 0 else 0
    st.metric(
        "Multi-Month Plans",
        f"{multi_month}",
        f"{multi_pct:.1f}% of total"
    )

with col8:
    if 'Days Until Renewal' in filtered_df.columns:
        upcoming_14 = len(filtered_df[filtered_df['Days Until Renewal'].between(0, 14, inclusive="both")])
        st.metric(
            "Renewals Due in 14 Days",
            f"{upcoming_14}"
        )

with col9:
    if 'Payment Status' in filtered_df.columns:
        unpaid = len(filtered_df[filtered_df['Payment Status'] == 'Unpaid / Zero'])
        unpaid_pct = (unpaid / total_students * 100) if total_students > 0 else 0
        st.metric(
            "Unpaid Records",
            f"{unpaid}",
            f"{unpaid_pct:.1f}%",
            delta_color="inverse"
        )

with col10:
    online_pmt = len(filtered_df[filtered_df['Mode of Payment'].str.contains('UPI|Bank Transfer', case=False, na=False)])
    online_pct = (online_pmt / total_students * 100) if total_students > 0 else 0
    st.metric(
        "Digital Payments",
        f"{online_pct:.1f}%",
        f"{online_pmt} transactions"
    )

# ============= MULTI-ACTIVITY INSIGHTS =============
st.markdown(
    '<div class="section-header"><h2>🎯 Multi-Activity & Admission Analysis</h2></div>',
    unsafe_allow_html=True
)

left_col, right_col = st.columns(2)

# ---------- LEFT: PARTICIPATION & REVENUE ----------
with left_col:
    st.subheader("Activity Participation & Revenue")

    c1, c2 = st.columns(2)

    # Activity bundle distribution
    with c1:
        st.markdown("**Activity Bundle Distribution**")
        if not filtered_df.empty:
            activity_count_dist = (
                filtered_df['Activity Count']
                .value_counts()
                .sort_index()
                .reset_index()
            )
            activity_count_dist.columns = ['Number of Activities', 'Students']

            fig_bundle = px.bar(
                activity_count_dist,
                x='Number of Activities',
                y='Students',
                text='Students',
            )
            fig_bundle.update_traces(textposition='outside')
            fig_bundle.update_layout(
                height=300,
                showlegend=False,
                xaxis_title="Number of Activities per Student",
                yaxis_title="Students"
            )
            st.plotly_chart(fig_bundle, use_container_width=True)

            avg_activities = filtered_df['Activity Count'].mean()
            st.caption(f"Average activities per student: **{avg_activities:.2f}**")

    # Revenue impact single vs multi
    with c2:
        st.markdown("**Revenue per Student: Single vs Multi-Activity**")
        if not filtered_df.empty:
            multi_comparison = (
                filtered_df
                .groupby('Is Multi-Activity')
                .agg({
                    'Name of Child': 'count',
                    'Amount': 'mean',
                    'Monthly Equivalent Fee': 'mean'
                })
                .reset_index()
            )
            multi_comparison['Is Multi-Activity'] = multi_comparison['Is Multi-Activity'].map({
                True: 'Multi-Activity',
                False: 'Single Activity'
            })
            multi_comparison.columns = [
                'Type',
                'Students',
                'Avg Total Fee',
                'Avg Monthly Fee'
            ]

            fig_multi_rev = go.Figure()
            fig_multi_rev.add_trace(go.Bar(
                name='Avg Package Value (₹)',
                x=multi_comparison['Type'],
                y=multi_comparison['Avg Total Fee'],
                marker_color='#3b82f6',
                text=multi_comparison['Avg Total Fee'].round(0),
                texttemplate='₹%{text:,.0f}',
                textposition='outside'
            ))
            fig_multi_rev.add_trace(go.Bar(
                name='Avg Monthly Fee (₹/month)',
                x=multi_comparison['Type'],
                y=multi_comparison['Avg Monthly Fee'],
                marker_color='#10b981',
                text=multi_comparison['Avg Monthly Fee'].round(0),
                texttemplate='₹%{text:,.0f}',
                textposition='outside'
            ))
            fig_multi_rev.update_layout(
                barmode='group',
                height=300,
                yaxis=dict(
                    title="Amount (₹)",
                    range=[
                        0,
                        multi_comparison[['Avg Total Fee', 'Avg Monthly Fee']].max().max() * 1.2
                    ]
                ),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0)
            )
            st.plotly_chart(fig_multi_rev, use_container_width=True)

            st.caption(
                "• **Avg Package Value (₹)** = average total fee collected per student for a package.\n\n"
                "• **Avg Monthly Fee (₹/month)** = average fee per student per month "
                "after dividing by package duration."
            )

# ---------- RIGHT: COMBINATIONS & ADMISSIONS ----------
with right_col:
    st.subheader("Popular Combinations & Admission Mix")

    c3, c4 = st.columns(2)

    # Popular activity combinations
    with c3:
        st.markdown("**Top Multi-Activity Combinations**")
        if not filtered_df.empty:
            multi_activity_df = filtered_df[filtered_df['Is Multi-Activity']]
            if not multi_activity_df.empty:
                combo_counts = (
                    multi_activity_df['Activity Opted']
                    .value_counts()
                    .head(5)
                    .reset_index()
                )
                combo_counts.columns = ['Activity Combination', 'Count']

                st.dataframe(
                    combo_counts.style.background_gradient(
                        subset=['Count'], cmap='YlOrRd'
                    ),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No multi-activity enrollments found.")

    # Admission type breakdown
    with c4:
        st.markdown("**Admission Type Breakdown**")
        if 'Type' in filtered_df.columns and not filtered_df.empty:
            type_dist = (
                filtered_df[filtered_df['Type'] != '']['Type']
                .value_counts()
                .reset_index()
            )
            type_dist.columns = ['Type', 'Count']

            type_colors = {
                'New Admission': '#10b981',
                'Renewal': '#3b82f6',
                'Re-Admission': '#f59e0b',
                'renewal': '#3b82f6',       # in case of lowercase labels
                're-admission': '#f59e0b'
            }

            fig_type = px.pie(
                type_dist,
                values='Count',
                names='Type',
                hole=0.45,
                color='Type',
                color_discrete_map=type_colors
            )
            fig_type.update_traces(textposition='inside', textinfo='percent+value')
            fig_type.update_layout(height=300, showlegend=True)
            st.plotly_chart(fig_type, use_container_width=True)

            total = type_dist['Count'].sum()
            for _, row in type_dist.iterrows():
                pct = (row['Count'] / total * 100) if total > 0 else 0
                st.caption(f"**{row['Type']}**: {row['Count']} ({pct:.1f}%)")
        else:
            st.info("Type column not available in data")

st.markdown("---")
st.markdown('<div class="section-header"><h2>📊 Individual Activity Performance</h2></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Activity Popularity (Individual Enrollments)")
    if not filtered_df_expanded.empty:
        activity_pop = filtered_df_expanded['Individual Activity'].value_counts().head(10).reset_index()
        activity_pop.columns = ['Activity', 'Enrollments']
        
        fig_pop = px.bar(
            activity_pop,
            x='Activity',
            y='Enrollments',
            text='Enrollments',
            color='Enrollments',
            color_continuous_scale='Blues'
        )
        fig_pop.update_traces(textposition='outside')
        fig_pop.update_layout(showlegend=False, xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig_pop, use_container_width=True)

with col2:
    st.subheader("Revenue by Individual Activity")
    if not filtered_df_expanded.empty:
        activity_rev = filtered_df_expanded.groupby('Individual Activity').agg({
            'Amount Per Activity': 'sum',
            'Monthly Fee Per Activity': 'sum',
            'Name of Child': 'count'
        }).reset_index()
        activity_rev.columns = ['Activity', 'Total Revenue', 'Monthly Revenue', 'Enrollments']
        activity_rev = activity_rev.sort_values('Monthly Revenue', ascending=False).head(10)
        
        fig_act_rev = go.Figure()
        fig_act_rev.add_trace(go.Bar(
            name='Total Revenue',
            x=activity_rev['Activity'],
            y=activity_rev['Total Revenue'],
            marker_color='#2563eb'
        ))
        fig_act_rev.add_trace(go.Bar(
            name='Monthly Revenue',
            x=activity_rev['Activity'],
            y=activity_rev['Monthly Revenue'],
            marker_color='#10b981'
        ))
        fig_act_rev.update_layout(barmode='group', xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig_act_rev, use_container_width=True)

# ============= RENEWAL PIPELINE =============
st.markdown("---")
st.markdown('<div class="section-header"><h2>🔄 Renewal Pipeline & Follow-ups</h2></div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    critical = filtered_df[(filtered_df['Retention Risk'] == 'Critical') & (filtered_df['Payment Status'] == 'Paid')]
    st.metric("🔴 Critical (Overdue)", len(critical), f"₹{critical['Due Amount'].sum():,.0f}")

with col2:
    high = filtered_df[(filtered_df['Retention Risk'] == 'High') & (filtered_df['Payment Status'] == 'Paid')]
    st.metric("🟡 High Risk (≤14d)", len(high))

with col3:
    active = filtered_df[(filtered_df['Retention Risk'] == 'Active') & (filtered_df['Payment Status'] == 'Paid')]
    st.metric("🟢 Active (>14d)", len(active))

with col4:
    unpaid_any = filtered_df[filtered_df['Payment Status'] == 'Unpaid / Zero']
    st.metric("⚪ Unpaid Records", len(unpaid_any))

# Priority Follow-up Table
st.markdown("#### 📋 Priority Follow-up List")

action_df = filtered_df[
    (filtered_df['Retention Risk'].isin(['Critical', 'High'])) | 
    (filtered_df['Payment Status'] == 'Unpaid / Zero')
].copy()
 

# --- Renewal reminder candidates: due soon and have phone number ---
REMINDER_WINDOW_DAYS = 7 # change to 14 etc if you want

if "Father's Number" in filtered_df.columns:
    reminder_df = action_df[
        (action_df['Days Until Renewal'].between(-500, REMINDER_WINDOW_DAYS, inclusive='both')) &
        (action_df["Father's Number"].str.strip() != "")
    ].copy()
else:
    reminder_df = pd.DataFrame()


if not action_df.empty:
    display_cols = [
        'Name of Child', 'Name of Parent', 'Activity Opted', 'Activity Count',
        'Payment Status', 'Amount', 'Monthly Equivalent Fee', 'Months Overdue', 'Due Amount', 'Date',
        'Fees Due Date', 'Renewal Date', 'Days Until Renewal', 'Retention Risk'
    ]
    if has_school:
        display_cols.insert(2, 'School')
    if has_class:
        display_cols.insert(3 if has_school else 2, 'Class')
    display_cols.append('Package Duration (Months)')
    display_cols.append('Mode of Payment')
    display_cols = [c for c in display_cols if c in action_df.columns]
    
    table_df = action_df[display_cols].sort_values(['Payment Status', 'Days Until Renewal'])
    
    # Format dates
    for date_col in ['Fees Due Date', 'Renewal Date', 'Date']:
        if date_col in table_df.columns:
            table_df[date_col] = pd.to_datetime(table_df[date_col]).dt.strftime('%d-%b-%Y')
    
    # Format currency
    if 'Amount' in table_df.columns:
        table_df['Amount'] = table_df['Amount'].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "")
    if 'Monthly Equivalent Fee' in table_df.columns:
        table_df['Monthly Equivalent Fee'] = table_df['Monthly Equivalent Fee'].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "")
    if 'Due Amount' in table_df.columns:
        table_df['Due Amount'] = table_df['Due Amount'].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "")
    
    def color_risk(val):
        if val == 'Critical':
            return 'background-color: #fecaca; font-weight: bold'
        elif val == 'High':
            return 'background-color: #fef3c7; font-weight: bold'
        elif val == 'Active':
            return 'background-color: #dcfce7;'
        return ''
    
    def color_payment(val):
        if val == 'Unpaid / Zero':
            return 'background-color: #fee2e2; font-weight:bold;'
        return ''
    
    def highlight_multi(val):
        if val > 1:
            return 'background-color: #fef3c7; font-weight: bold;'
        return ''
    
    styled_df = table_df.style
    if 'Retention Risk' in table_df.columns:
        styled_df = styled_df.applymap(color_risk, subset=['Retention Risk'])
    if 'Payment Status' in table_df.columns:
        styled_df = styled_df.applymap(color_payment, subset=['Payment Status'])
    if 'Activity Count' in table_df.columns:
        styled_df = styled_df.applymap(highlight_multi, subset=['Activity Count'])
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    critical_count = len(action_df[action_df['Retention Risk'] == 'Critical'])
    high_count = len(action_df[action_df['Retention Risk'] == 'High'])
    unpaid_count = len(action_df[action_df['Payment Status'] == 'Unpaid / Zero'])
    multi_activity_risk = len(action_df[action_df['Is Multi-Activity']])
    
    # Calculate average months overdue for critical students
    critical_students = action_df[action_df['Retention Risk'] == 'Critical']
    if len(critical_students) > 0 and 'Months Overdue' in critical_students.columns:
        avg_months_overdue = critical_students['Months Overdue'].mean()
        total_overdue_amount = critical_students['Due Amount'].sum()
        overdue_text = f"₹{total_overdue_amount:,.0f} (avg {avg_months_overdue:.1f} months overdue)"
    else:
        overdue_text = "₹0"
    
    st.warning(f"""
    **📞 Action Summary**
    - **{critical_count} Critical** paid customers: overdue and need immediate contact ({overdue_text})
    - **{high_count} High-risk** paid customers: due within the next 14 days  
    - **{unpaid_count} Unpaid / Zero** records: payment not received yet, prioritize follow-up  
    - **{multi_activity_risk} Multi-activity** students in this list (higher value retention)
    - Monthly revenue at risk (paid customers): ₹{action_df[action_df['Payment Status'] == 'Paid']['Monthly Equivalent Fee'].sum():,.2f}
    """)
    
    csv = table_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Follow-up List",
        data=csv,
        file_name=f'action_required_{datetime.now().strftime("%Y%m%d")}.csv',
        mime='text/csv'
    )
    st.markdown("#### 📲 Renewal Reminder Messages")

    if not reminder_df.empty:
        # Build a friendly message for each parent
        def build_reminder(row):
            parent = row.get('Name of Parent') or "Parent"
            child = row.get('Name of Child') or "your child"
            activity = row.get('Activity Opted') or "the enrolled activity"
            renew_date = pd.to_datetime(row['Renewal Date']).strftime('%d-%b-%Y') if pd.notna(row['Renewal Date']) else "soon"
            amount = row.get('Monthly Equivalent Fee', 0)
            amount_txt = f"₹{amount:,.0f}" if pd.notna(amount) and amount > 0 else "the fees"

            return (
                f"Dear {parent}, this is a reminder from your activity centre. "
                f"{child}'s {activity} package is due for renewal on {renew_date}. "
                f"Please pay {amount_txt} to ensure uninterrupted classes. "
                f"Thank you!"
            )

        reminder_df['Reminder Message'] = reminder_df.apply(build_reminder, axis=1)

        # Columns to show / export
        preview_cols = ["Father's Number", 'Name of Parent', 'Name of Child',
                        'Activity Opted', 'Renewal Date', 'Days Until Renewal', 'Reminder Message']
        preview_cols = [c for c in preview_cols if c in reminder_df.columns]

        st.dataframe(
            reminder_df[preview_cols].head(20),
            use_container_width=True,
            hide_index=True
        )

        # CSV for bulk-SMS / WhatsApp upload
        reminder_csv = reminder_df[preview_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Reminder List (Phone + Message)",
            data=reminder_csv,
            file_name=f'renewal_reminders_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv'
        )

        st.caption(
            f"Includes only students with renewal due in the next {REMINDER_WINDOW_DAYS} days "
            "and a valid father's mobile number."
        )
    else:
        st.info(
            f"No students with a renewal due in the next {REMINDER_WINDOW_DAYS} days "
            "and a valid father's number, based on current filters."
        )
else:
    st.success("✅ No immediate follow-up required based on current filters.")
# ============= DETAILED ANALYTICS TABS =============
st.markdown("---")
st.markdown('<div class="section-header"><h2>📈 Detailed Analytics & Insights</h2></div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "💰 Financial Trends",
    "💳 Payment Methods",
    "🏫 School Analysis",
    "📚 Class Distribution",
    "📦 Package Analysis",
    "⏱️ Payment Timing",
    "📌 Business Insights",
])


with tab1:
    st.subheader("📈 Monthly Revenue & Growth Trends")
    
    # Revenue recognition method selector
    col_method, col_spacer = st.columns([1, 2])
    with col_method:
        revenue_method = st.radio(
            "Revenue Recognition:",
            ["💰 Cash Basis", "📊 Accrual Basis"],
            help="Cash: Full payment shown when received. Accrual: Spread across package months."
        )
    
    if not filtered_df.empty:
        trend_df = filtered_df.copy()
        
        if revenue_method == "📊 Accrual Basis":
            # Accrual method: Spread revenue across package duration
            st.info("""
            **📊 Accrual Basis (Subscription Revenue Model)**
            - Multi-month packages are spread evenly across their duration
            - Shows true Monthly Recurring Revenue (MRR)
            - Example: ₹6,000 for 6 months = ₹1,000 recognized each month
            - Best for understanding business stability and growth trends
            """)
            
            accrual_rows = []
            for idx, row in trend_df.iterrows():
                package_months = int(row['Package Duration (Months)'])
                monthly_revenue = row['Monthly Equivalent Fee']
                payment_date = row['Date']
                
                for month_offset in range(package_months):
                    new_row = {
                        'Month': (payment_date + pd.DateOffset(months=month_offset)).to_period('M').to_timestamp(),
                        'Revenue': monthly_revenue,
                        'Type': row.get('Type', 'Unknown'),
                        'Is Multi-Activity': row['Is Multi-Activity'],
                        'Student': row['Name of Child'],
                        'Is New': 1 if month_offset == 0 else 0  # Only count as new in first month
                    }
                    accrual_rows.append(new_row)
            
            accrual_df = pd.DataFrame(accrual_rows)
            
            # Aggregate by month and type
            monthly = accrual_df.groupby('Month').agg(
                Total_Revenue=('Revenue', 'sum'),
                Active_Students=('Student', 'nunique')
            ).reset_index()
            
            # Get new admissions and renewals by month
            if 'Type' in trend_df.columns:
                type_monthly = trend_df.copy()
                type_monthly['Month'] = type_monthly['Date'].dt.to_period('M').dt.to_timestamp()
                type_summary = type_monthly.groupby(['Month', 'Type']).size().reset_index(name='Count')
                type_pivot = type_summary.pivot(index='Month', columns='Type', values='Count').fillna(0).reset_index()
                monthly = monthly.merge(type_pivot, on='Month', how='left').fillna(0)
            
        else:  # Cash Basis
            st.success("""
            **💰 Cash Basis (Actual Cash Flow)**
            - Full payment amount shown in the month it was received
            - Reflects actual money in the bank
            - Example: ₹6,000 for 6 months = ₹6,000 in payment month, then ₹0
            - Best for cash flow planning and financial reporting
            """)
            
            trend_df['Month'] = trend_df['Date'].dt.to_period('M').dt.to_timestamp()
            
            monthly = trend_df.groupby('Month').agg(
                Total_Revenue=('Amount', 'sum'),
                Active_Students=('Name of Child', 'nunique')
            ).reset_index()
            
            # Get breakdown by type
            if 'Type' in trend_df.columns:
                type_summary = trend_df.groupby(['Month', 'Type']).agg(
                    Count=('Name of Child', 'count'),
                    Revenue=('Amount', 'sum')
                ).reset_index()
                
                # Pivot for counts
                type_count_pivot = type_summary.pivot(index='Month', columns='Type', values='Count').fillna(0).reset_index()
                monthly = monthly.merge(type_count_pivot, on='Month', how='left').fillna(0)
                
                # Pivot for revenue
                type_rev_pivot = type_summary.pivot(index='Month', columns='Type', values='Revenue').fillna(0).reset_index()
                type_rev_pivot.columns = [f'{col}_Revenue' if col != 'Month' else col for col in type_rev_pivot.columns]
                monthly = monthly.merge(type_rev_pivot, on='Month', how='left').fillna(0)
        
        if not monthly.empty:
            # Create visualization
                        # Create visualization
            fig_trend = go.Figure()

            # Add revenue bars by type if available
            if 'Type' in trend_df.columns:
                if revenue_method == "💰 Cash Basis":
                    # Use only *_Revenue columns (e.g. 'renewal_Revenue')
                    revenue_cols = [c for c in monthly.columns if c.endswith('_Revenue')]

                    # Explicit colors (keyed by lowercase type name)
                    base_color_map = {
                        'new admission': '#10b981',
                        'renewal': '#3b82f6',
                        're-admission': '#f59e0b',
                    }
                    # Fallback palette for any extra types
                    fallback_palette = px.colors.qualitative.Plotly

                    # Stacked bars by admission type
                    for i, col in enumerate(revenue_cols):
                        adm_type = col.replace('_Revenue', '')           # original label (e.g. 'renewal')
                        key = adm_type.strip().lower()                  # normalised for color lookup
                        color = base_color_map.get(key, fallback_palette[i % len(fallback_palette)])

                        fig_trend.add_trace(go.Bar(
                            x=monthly['Month'],
                            y=monthly[col],
                            name=adm_type,
                            marker_color=color,
                            yaxis='y1'
                        ))
                else:
                    # Accrual basis: show total MRR as a single bar
                    fig_trend.add_trace(go.Bar(
                        x=monthly['Month'],
                        y=monthly['Total_Revenue'],
                        name='MRR',
                        marker_color='#8b5cf6',
                        yaxis='y1'
                    ))
            else:
                fig_trend.add_trace(go.Bar(
                    x=monthly['Month'],
                    y=monthly['Total_Revenue'],
                    name='Revenue',
                    marker_color='#3b82f6',
                    yaxis='y1'
                ))

            
            # Add active students line
            fig_trend.add_trace(go.Scatter(
                x=monthly['Month'],
                y=monthly['Active_Students'],
                name='Active Students',
                mode='lines+markers',
                yaxis='y2',
                line=dict(color='#ef4444', width=3),
                marker=dict(size=8)
            ))
            
            fig_trend.update_layout(
                xaxis=dict(title="Month"),
                yaxis=dict(title="Revenue (₹)", side='left'),
                yaxis2=dict(title="Students", overlaying='y', side='right'),
                legend=dict(orientation='h', yanchor='bottom', y=1.02),
                height=500,
                barmode='stack',
                hovermode='x unified'
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                total_rev = monthly['Total_Revenue'].sum()
                st.metric("Total Revenue", f"₹{total_rev:,.0f}")
            with col2:
                avg_monthly_rev = monthly['Total_Revenue'].mean()
                st.metric("Avg Monthly Revenue", f"₹{avg_monthly_rev:,.0f}")
            with col3:
                if 'Type' in trend_df.columns and 'New Admission' in monthly.columns:
                    total_new = monthly['New Admission'].sum() if revenue_method == "💰 Cash Basis" else trend_df[trend_df['Type'] == 'New Admission'].shape[0]
                    st.metric("Total New Admissions", f"{int(total_new)}")
                else:
                    st.metric("Total Payments", f"{len(trend_df)}")
            with col4:
                if 'Type' in trend_df.columns and 'Renewal' in monthly.columns:
                    total_renewals = monthly['Renewal'].sum() if revenue_method == "💰 Cash Basis" else trend_df[trend_df['Type'] == 'Renewal'].shape[0]
                    renewal_rate = (total_renewals / len(trend_df) * 100) if len(trend_df) > 0 else 0
                    st.metric("Renewal Rate", f"{renewal_rate:.1f}%", f"{int(total_renewals)} renewals")
            
            # Detailed breakdown by admission type
            if 'Type' in trend_df.columns:
                st.markdown("---")
                st.markdown("#### 📊 Revenue Breakdown by Admission Type")
                
                type_breakdown = trend_df.groupby('Type').agg({
                    'Name of Child': 'count',
                    'Amount': 'sum',
                    'Monthly Equivalent Fee': 'sum'
                }).reset_index()
                type_breakdown.columns = ['Type', 'Count', 'Total Cash Collected', 'Total MRR']
                type_breakdown['Avg Payment'] = (type_breakdown['Total Cash Collected'] / type_breakdown['Count']).round(0)
                
                col_a, col_b = st.columns([2, 1])
                
                with col_a:
                    fig_type_rev = px.bar(
                        type_breakdown,
                        x='Type',
                        y='Total Cash Collected',
                        text='Total Cash Collected',
                        color='Type',
                        color_discrete_map={
                            'New Admission': '#10b981',
                            'Renewal': '#3b82f6',
                            'Re-Admission': '#f59e0b'
                        }
                    )
                    fig_type_rev.update_traces(
                        texttemplate='₹%{text:,.0f}',
                        textposition='outside'
                    )
                    fig_type_rev.update_layout(
                        showlegend=False,
                        height=350,
                        yaxis=dict(range=[0, type_breakdown['Total Cash Collected'].max() * 1.15])
                    )
                    st.plotly_chart(fig_type_rev, use_container_width=True)
                
                with col_b:
                    st.dataframe(
                        type_breakdown.style.format({
                            'Count': '{:,.0f}',
                            'Total Cash Collected': '₹{:,.0f}',
                            'Total MRR': '₹{:,.0f}',
                            'Avg Payment': '₹{:,.0f}'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
            
            # Show monthly data table
            with st.expander("📄 View Month-by-Month Data"):
                display_monthly = monthly.copy()
                display_monthly['Month'] = display_monthly['Month'].dt.strftime('%b %Y')
                display_monthly['Total_Revenue'] = display_monthly['Total_Revenue'].apply(lambda x: f"₹{x:,.0f}")
                
                st.dataframe(
                    display_monthly,
                    use_container_width=True,
                    hide_index=True
                )
    else:
        st.info("No data available for the selected filters.")

with tab2:
    st.subheader("💳 Payment Method Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Payment Method Distribution")
        pmt_dist = filtered_df[filtered_df['Mode of Payment'] != '']['Mode of Payment'].value_counts().reset_index()
        pmt_dist.columns = ['Method', 'Count']
        
        fig_pmt_pie = px.pie(
            pmt_dist,
            values='Count',
            names='Method',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_pmt_pie.update_traces(textposition='inside', textinfo='percent+label+value')
        fig_pmt_pie.update_layout(height=400)
        st.plotly_chart(fig_pmt_pie, use_container_width=True)
    
    with col2:
        st.markdown("##### Revenue by Payment Method")
        pmt_rev = filtered_df[filtered_df['Mode of Payment'] != ''].groupby('Mode of Payment').agg({
            'Amount': 'sum',
            'Monthly Equivalent Fee': 'sum',
            'Name of Child': 'count'
        }).reset_index()
        pmt_rev.columns = ['Payment Method', 'Total Revenue', 'Monthly Revenue', 'Transactions']
        pmt_rev = pmt_rev.sort_values('Total Revenue', ascending=False)
        
        fig_pmt_bar = px.bar(
            pmt_rev,
            x='Payment Method',
            y='Total Revenue',
            text='Total Revenue',
            color='Payment Method',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pmt_bar.update_traces(
            texttemplate='₹%{text:,.0f}',
            textposition='outside'
        )
        fig_pmt_bar.update_layout(
            showlegend=False,
            xaxis_tickangle=-45,
            height=400,
            yaxis=dict(range=[0, pmt_rev['Total Revenue'].max() * 1.15])
        )
        st.plotly_chart(fig_pmt_bar, use_container_width=True)
    
    # Payment Method Details Table
    st.markdown("##### Detailed Payment Method Breakdown")
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        digital_methods = ['UPI', 'Bank Transfer', 'Online', 'Card', 'Paytm', 'PhonePe', 'GPay']
        digital_count = filtered_df[filtered_df['Mode of Payment'].str.contains('|'.join(digital_methods), case=False, na=False)]['Name of Child'].count()
        digital_revenue = filtered_df[filtered_df['Mode of Payment'].str.contains('|'.join(digital_methods), case=False, na=False)]['Amount'].sum()
        st.metric("Digital Payments", f"{digital_count}", f"₹{digital_revenue:,.0f}")
    
    with col_b:
        cash_count = filtered_df[filtered_df['Mode of Payment'].str.contains('Cash', case=False, na=False)]['Name of Child'].count()
        cash_revenue = filtered_df[filtered_df['Mode of Payment'].str.contains('Cash', case=False, na=False)]['Amount'].sum()
        st.metric("Cash Payments", f"{cash_count}", f"₹{cash_revenue:,.0f}")
    
    with col_c:
        avg_digital = digital_revenue / digital_count if digital_count > 0 else 0
        avg_cash = cash_revenue / cash_count if cash_count > 0 else 0
        st.metric("Avg Digital Transaction", f"₹{avg_digital:,.0f}", f"vs Cash: ₹{avg_cash:,.0f}")
    
    # Payment method by activity
    st.markdown("##### Payment Method Preference by Activity")
    if not filtered_df_expanded.empty:
        pmt_activity = filtered_df_expanded.groupby(['Individual Activity', 'Mode of Payment']).size().reset_index(name='Count')
        pmt_activity_pivot = pmt_activity.pivot(index='Individual Activity', columns='Mode of Payment', values='Count').fillna(0)
        
        top_activities = filtered_df_expanded['Individual Activity'].value_counts().head(5).index
        pmt_activity_top = pmt_activity[pmt_activity['Individual Activity'].isin(top_activities)]
        
        fig_pmt_act = px.bar(
            pmt_activity_top,
            x='Individual Activity',
            y='Count',
            color='Mode of Payment',
            barmode='stack',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_pmt_act.update_layout(
            xaxis_tickangle=-45,
            height=350,
            legend=dict(orientation='h', yanchor='bottom', y=-0.3)
        )
        st.plotly_chart(fig_pmt_act, use_container_width=True)
    
    # Detailed table
    st.dataframe(
        pmt_rev.style.format({
            'Total Revenue': '₹{:,.0f}',
            'Monthly Revenue': '₹{:,.0f}',
            'Transactions': '{:,.0f}'
        }).background_gradient(subset=['Total Revenue'], cmap='Greens'),
        use_container_width=True,
        hide_index=True
    )

with tab3:
    if has_school:
        st.subheader("School-wise Performance")
        school_analysis = filtered_df[filtered_df['School'] != ''].groupby('School').agg({
            'Name of Child': 'count',
            'Amount': 'sum',
            'Monthly Equivalent Fee': 'sum',
            'Is Multi-Activity': 'sum',
            'Activity Count': 'sum'
        }).round(2)
        if not school_analysis.empty:
            school_analysis.columns = ['Students', 'Total Revenue', 'Monthly Revenue', 'Multi-Activity Students', 'Total Activity Enrollments']
            school_analysis['Avg Activities per Student'] = (school_analysis['Total Activity Enrollments'] / school_analysis['Students']).round(2)
            school_analysis = school_analysis.sort_values('Students', ascending=False)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                fig_school = px.bar(
                    school_analysis.reset_index(),
                    x='School',
                    y=['Students', 'Multi-Activity Students'],
                    barmode='group',
                    labels={'value': 'Count', 'variable': 'Type'}
                )
                fig_school.update_layout(xaxis_tickangle=-45, height=400)
                st.plotly_chart(fig_school, use_container_width=True)
            
            with col2:
                st.dataframe(
                    school_analysis[['Students', 'Multi-Activity Students', 'Avg Activities per Student', 'Monthly Revenue']].style.format({
                        'Students': '{:.0f}',
                        'Multi-Activity Students': '{:.0f}',
                        'Avg Activities per Student': '{:.2f}',
                        'Monthly Revenue': '₹{:,.0f}'
                    }),
                    use_container_width=True
                )
    else:
        st.info("School data not available.")

with tab4:
    if has_class:
        st.subheader("Class-wise Distribution & Multi-Activity Adoption")
        class_dist = filtered_df[filtered_df['Class'] != ''].groupby('Class').agg({
            'Name of Child': 'count',
            'Monthly Equivalent Fee': 'mean',
            'Is Multi-Activity': 'sum',
            'Activity Count': 'mean'
        }).round(2)
        if not class_dist.empty:
            class_dist.columns = ['Students', 'Avg Monthly Fee', 'Multi-Activity Count', 'Avg Activities']
            class_dist = class_dist.sort_values('Students', ascending=False)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                fig_class = px.bar(
                    class_dist.reset_index(),
                    x='Class',
                    y='Students',
                    color='Avg Activities',
                    color_continuous_scale='Viridis',
                    text='Students'
                )
                fig_class.update_traces(textposition='outside')
                st.plotly_chart(fig_class, use_container_width=True)
            
            with col2:
                st.dataframe(
                    class_dist.style.format({
                        'Students': '{:.0f}',
                        'Avg Monthly Fee': '₹{:,.0f}',
                        'Multi-Activity Count': '{:.0f}',
                        'Avg Activities': '{:.2f}'
                    }),
                    use_container_width=True
                )
    else:
        st.info("Class data not available.")

with tab5:
    st.subheader("Package Duration Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Package Distribution")
        pkg_dist = filtered_df['Package Duration (Months)'].value_counts().reset_index()
        pkg_dist.columns = ['Duration', 'Count']
        pkg_map = {1: '1 Month', 2: '2 Months', 3: '3 Months', 4: '4 Months', 
                   5: '5 Months', 6: '6 Months', 12: 'Yearly'}
        pkg_dist['Duration Label'] = pkg_dist['Duration'].map(lambda x: pkg_map.get(x, f'{x} Months'))
        
        fig3 = px.pie(
            pkg_dist,
            values='Count',
            names='Duration Label',
            hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig3.update_traces(textposition='inside', textinfo='percent+label')
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.markdown("##### Multi-Activity by Package Duration")
        pkg_multi = filtered_df.groupby('Package Duration (Months)').agg({
            'Name of Child': 'count',
            'Is Multi-Activity': 'sum'
        }).reset_index()
        pkg_multi.columns = ['Duration', 'Total Students', 'Multi-Activity']
        pkg_multi['Multi-Activity %'] = (pkg_multi['Multi-Activity'] / pkg_multi['Total Students'] * 100).round(1)
        pkg_multi['Duration Label'] = pkg_multi['Duration'].map(lambda x: pkg_map.get(x, f'{x}M'))
        
        fig_pkg_multi = px.bar(
            pkg_multi,
            x='Duration Label',
            y=['Total Students', 'Multi-Activity'],
            barmode='group',
            text_auto=True
        )
        fig_pkg_multi.update_layout(height=400)
        st.plotly_chart(fig_pkg_multi, use_container_width=True)

with tab6:
    st.subheader("Payment Timeline Analysis")
    
    if not filtered_df.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_gap = filtered_df['Days Since Payment'].mean()
            st.metric("Avg Days Since Last Payment", f"{avg_gap:.1f} days")
        
        with col2:
            if 'Days Until Fees Due' in filtered_df.columns:
                avg_until_due = filtered_df[filtered_df['Days Until Fees Due'] > 0]['Days Until Fees Due'].mean()
                st.metric("Avg Days Until Fees Due", f"{avg_until_due:.1f} days" if not pd.isna(avg_until_due) else "N/A")
        
        with col3:
            if 'Days Until Renewal' in filtered_df.columns:
                avg_until_renewal = filtered_df[filtered_df['Days Until Renewal'] > 0]['Days Until Renewal'].mean()
                st.metric("Avg Days Until Renewal", f"{avg_until_renewal:.1f} days" if not pd.isna(avg_until_renewal) else "N/A")
        
        st.markdown("##### Payment Age Distribution")
        col_a, col_b = st.columns(2)
        
        with col_a:
            age_df = filtered_df.dropna(subset=['Payment Age Bucket'])
            if not age_df.empty:
                age_dist = age_df['Payment Age Bucket'].value_counts().reset_index()
                age_dist.columns = ['Age Bucket', 'Count']
                
                fig_age = px.bar(
                    age_dist,
                    x='Age Bucket',
                    y='Count',
                    text='Count',
                    color='Count',
                    color_continuous_scale='Blues'
                )
                fig_age.update_traces(textposition='outside')
                fig_age.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig_age, use_container_width=True)
        
        with col_b:
            st.markdown("##### Renewal Timeline Distribution")
            renewal_bins = pd.cut(
                filtered_df['Days Until Renewal'],
                bins=[-float('inf'), 0, 14, 30, 60, float('inf')],
                labels=['Overdue', '0-14 days', '15-30 days', '31-60 days', '60+ days']
            )
            renewal_dist = renewal_bins.value_counts().reset_index()
            renewal_dist.columns = ['Timeline', 'Count']
            
            fig_renewal = px.bar(
                renewal_dist,
                x='Timeline',
                y='Count',
                text='Count',
                color='Timeline',
                color_discrete_map={
                    'Overdue': '#ef4444',
                    '0-14 days': '#f59e0b',
                    '15-30 days': '#eab308',
                    '31-60 days': '#10b981',
                    '60+ days': '#3b82f6'
                }
            )
            fig_renewal.update_traces(textposition='outside')
            fig_renewal.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_renewal, use_container_width=True)

with tab7:
    st.subheader("Business Insights & Recommendations")

    # Key Findings
    st.markdown("### 🔍 Key Findings")
    findings = []
    
    if len(filtered_df) > 0:
        # Multi-activity insights
        multi_pct = (len(filtered_df[filtered_df['Is Multi-Activity']]) / len(filtered_df) * 100)
        findings.append(f"- **{multi_pct:.1f}%** of students are enrolled in multiple activities")
        
        if multi_pct > 0:
            multi_rev = filtered_df[filtered_df['Is Multi-Activity']]['Monthly Equivalent Fee'].mean()
            single_rev = filtered_df[~filtered_df['Is Multi-Activity']]['Monthly Equivalent Fee'].mean()
            if not pd.isna(multi_rev) and not pd.isna(single_rev):
                uplift = ((multi_rev - single_rev) / single_rev * 100) if single_rev > 0 else 0
                findings.append(f"- Multi-activity students generate **{uplift:+.1f}%** more monthly revenue on average")
        
        # Admission type insights
        if 'Type' in filtered_df.columns:
            type_counts = filtered_df['Type'].value_counts()
            total_count = len(filtered_df)
            
            if 'New Admission' in type_counts:
                new_pct = (type_counts['New Admission'] / total_count * 100)
                findings.append(f"- **{new_pct:.1f}%** are new admissions ({type_counts['New Admission']} students)")
            
            if 'Renewal' in type_counts:
                renewal_pct = (type_counts['Renewal'] / total_count * 100)
                findings.append(f"- **{renewal_pct:.1f}%** are renewals ({type_counts['Renewal']} students)")
                
                # Renewal revenue comparison
                renewal_rev = filtered_df[filtered_df['Type'] == 'Renewal']['Amount'].mean()
                new_rev = filtered_df[filtered_df['Type'] == 'New Admission']['Amount'].mean() if 'New Admission' in type_counts else 0
                if not pd.isna(renewal_rev) and not pd.isna(new_rev) and new_rev > 0:
                    rev_diff = ((renewal_rev - new_rev) / new_rev * 100)
                    findings.append(f"- Renewal customers pay **{rev_diff:+.1f}%** {'more' if rev_diff > 0 else 'less'} on average than new admissions")
            
            if 'Re-Admission' in type_counts:
                readm_pct = (type_counts['Re-Admission'] / total_count * 100)
                findings.append(f"- **{readm_pct:.1f}%** are re-admissions ({type_counts['Re-Admission']} students) - focus on retention!")
        
        # Package duration insights
        long_pkg = len(filtered_df[filtered_df['Package Duration (Months)'] >= 6]) / len(filtered_df) * 100
        findings.append(f"- **{long_pkg:.1f}%** of students have 6+ month packages")
        
        # Package by admission type
        if 'Type' in filtered_df.columns:
            new_long_pkg = len(filtered_df[(filtered_df['Type'] == 'New Admission') & (filtered_df['Package Duration (Months)'] >= 6)])
            renewal_long_pkg = len(filtered_df[(filtered_df['Type'] == 'Renewal') & (filtered_df['Package Duration (Months)'] >= 6)])
            
            if 'New Admission' in type_counts and type_counts['New Admission'] > 0:
                new_long_pct = (new_long_pkg / type_counts['New Admission'] * 100)
                findings.append(f"  - Only **{new_long_pct:.1f}%** of new admissions choose long-term packages")
            
            if 'Renewal' in type_counts and type_counts['Renewal'] > 0:
                renewal_long_pct = (renewal_long_pkg / type_counts['Renewal'] * 100)
                findings.append(f"  - **{renewal_long_pct:.1f}%** of renewals opt for long-term packages")
        
        # Digital payment insights
        digital_pmt = len(filtered_df[filtered_df['Mode of Payment'].str.contains('UPI|Bank Transfer', case=False, na=False)]) / len(filtered_df) * 100
        findings.append(f"- **{digital_pmt:.1f}%** of payments are digital (UPI/Bank Transfer)")
    
    for f in findings:
        st.markdown(f)
    
    # Strategic Recommendations
    st.markdown("### 🎯 Strategic Recommendations")
    recs = []
    
    if len(filtered_df) > 0:
        # Multi-activity recommendations
        if multi_pct < 25:
            recs.append("**1. Promote Activity Bundling:** Only {:.1f}% of students take multiple activities. Create combo packages with 10-15% discounts to increase cross-enrollment.".format(multi_pct))
        elif multi_pct > 40:
            recs.append("**1. Optimize Multi-Activity Experience:** With {:.1f}% multi-activity adoption, focus on schedule coordination and parent communication for these high-value students.".format(multi_pct))
        
        # Admission type recommendations
        if 'Type' in filtered_df.columns:
            type_counts = filtered_df['Type'].value_counts()
            
            if 'Re-Admission' in type_counts:
                readm_count = type_counts['Re-Admission']
                if readm_count > 0:
                    readm_pct = (readm_count / len(filtered_df) * 100)
                    if readm_pct > 10:
                        recs.append(f"**2. Reduce Churn:** {readm_pct:.1f}% are re-admissions (students who left and came back). Investigate why students are leaving and implement better retention strategies.")
            
            if 'Renewal' in type_counts and 'New Admission' in type_counts:
                renewal_rate = (type_counts['Renewal'] / (type_counts['Renewal'] + type_counts.get('Re-Admission', 0)) * 100)
                if renewal_rate < 70:
                    recs.append(f"**3. Improve Renewal Process:** Current renewal rate is {renewal_rate:.1f}%. Implement proactive renewal campaigns starting 30 days before expiry with special offers.")
            
            # Long-term package recommendations by type
            if 'New Admission' in type_counts and type_counts['New Admission'] > 0:
                new_long_pkg = len(filtered_df[(filtered_df['Type'] == 'New Admission') & (filtered_df['Package Duration (Months)'] >= 6)])
                new_long_pct = (new_long_pkg / type_counts['New Admission'] * 100)
                if new_long_pct < 20:
                    recs.append(f"**4. Upsell New Admissions:** Only {new_long_pct:.1f}% of new students choose 6+ month packages. Offer first-time signup bonuses for annual packages.")
        
        # Critical overdue recommendations
        critical_rate = len(filtered_df[filtered_df['Retention Risk'] == 'Critical']) / len(filtered_df) * 100
        if critical_rate > 10:
            recs.append(f"**5. Urgent: Strengthen Retention Process:** {critical_rate:.1f}% of paid students are overdue. Implement automated reminders at D-7, D-3, and D+1.")
        
        # Digital payment recommendations
        if digital_pmt < 50:
            recs.append("**6. Accelerate Digital Adoption:** Share UPI QR codes and payment links proactively. Consider small incentives for digital payments.")
        
        # Activity-specific insights
        if not filtered_df_expanded.empty:
            top_activity = filtered_df_expanded['Individual Activity'].value_counts().index[0]
            recs.append(f"**7. Leverage Top Performer:** '{top_activity}' is your most popular activity. Use it as an anchor for combo packages and cross-selling.")
    
    if recs:
        for r in recs:
            st.markdown(f"- {r}")
    else:
        st.info("Not enough data to generate recommendations.")
    
    # Growth opportunities section
    if 'Type' in filtered_df.columns and len(filtered_df) > 0:
        st.markdown("### 💡 Growth Opportunities")
        opportunities = []
        
        type_counts = filtered_df['Type'].value_counts()
        
        # Calculate customer lifetime value by type
        if 'Renewal' in type_counts and type_counts['Renewal'] > 0:
            renewal_students = filtered_df[filtered_df['Type'] == 'Renewal']
            avg_renewal_value = renewal_students['Amount'].mean()
            opportunities.append(f"- **High-Value Renewals**: Average renewal value is ₹{avg_renewal_value:,.0f}. Focus retention efforts on students nearing their expiry date.")
        
        # Re-admission recovery
        if 'Re-Admission' in type_counts and type_counts['Re-Admission'] > 0:
            opportunities.append(f"- **Win-Back Campaign**: {type_counts['Re-Admission']} students re-admitted. Create a systematic win-back campaign for churned students with special offers.")
        
        # Multi-activity upsell
        single_activity_count = len(filtered_df[~filtered_df['Is Multi-Activity']])
        if single_activity_count > 0:
            potential_revenue = single_activity_count * filtered_df['Monthly Equivalent Fee'].mean() * 0.5  # Assuming 50% uplift
            opportunities.append(f"- **Multi-Activity Upsell**: {single_activity_count} students have single activities. Converting 20% to multi-activity could add ₹{potential_revenue * 0.2:,.0f}/month in MRR.")
        
        for opp in opportunities:
            st.markdown(opp)

# ============= EXPORT & FOOTER =============
st.markdown("---")
st.subheader("📥 Export Reports")

col1, col2, col3, col4 = st.columns(4)

with col1:
    full_csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Full Report (Filtered)",
        data=full_csv,
        file_name=f'enrollment_full_{datetime.now().strftime("%Y%m%d")}.csv',
        mime='text/csv'
    )

with col2:
    if 'Retention Risk' in filtered_df.columns:
        critical_csv = filtered_df[
            (filtered_df['Retention Risk'] == 'Critical') & (filtered_df['Payment Status'] == 'Paid')
        ].to_csv(index=False).encode('utf-8')
        st.download_button(
            "Overdue Paid List",
            data=critical_csv,
            file_name=f'overdue_paid_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv'
        )

with col3:
    multi_activity_csv = filtered_df[filtered_df['Is Multi-Activity']].to_csv(index=False).encode('utf-8')
    st.download_button(
        "Multi-Activity Students",
        data=multi_activity_csv,
        file_name=f'multi_activity_{datetime.now().strftime("%Y%m%d")}.csv',
        mime='text/csv'
    )

with col4:
    if not filtered_df_expanded.empty:
        activity_breakdown_csv = filtered_df_expanded[['Name of Child', 'Individual Activity', 'Amount Per Activity', 'Monthly Fee Per Activity']].to_csv(index=False).encode('utf-8')
        st.download_button(
            "Activity Breakdown",
            data=activity_breakdown_csv,
            file_name=f'activity_breakdown_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv'
        )

# Footer
st.markdown("---")
today_ref = df['Date'].max() if not df['Date'].isna().all() else pd.Timestamp.now()
st.markdown(f"""
<div style='text-align: center; color: #6b7280; padding: 20px;'>
    <p style='font-size: 1.05em;'><strong>Activity Enrollment Management System v2.0</strong></p>
    <p>Dashboard Updated: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}</p>
    <p>Reference Date (Latest Payment): {today_ref.strftime("%B %d, %Y")}</p>
    <p>Total Records: {len(df):,} | Filtered Records: {len(filtered_df):,} | Total Activity Enrollments: {df['Activity Count'].sum():.0f}</p>
    <p style='font-size: 0.9em; margin-top: 10px;'>
        <em>🎯 Focus Areas: Multi-activity bundling • Unpaid follow-ups • Critical renewals • Long-term package adoption</em>
    </p>
</div>
""", unsafe_allow_html=True)
