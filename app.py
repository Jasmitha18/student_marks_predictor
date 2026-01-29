import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# =====================================================
# SESSION STATE
# =====================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "teacher" not in st.session_state:
    st.session_state.teacher = None

# =====================================================
# FILES
# =====================================================
teachers_file = "teachers.csv"
history_file = "prediction_history.csv"

# =====================================================
# LOAD / CREATE TEACHERS
# =====================================================
try:
    teachers_df = pd.read_csv(teachers_file)
except:
    teachers_df = pd.DataFrame(
        columns=["username", "password", "name", "subject", "email"]
    )
    teachers_df.to_csv(teachers_file, index=False)

# =====================================================
# LOAD / CREATE HISTORY
# =====================================================
try:
    history_df = pd.read_csv(history_file)
except:
    history_df = pd.DataFrame(columns=[
        "teacher_username",
        "student_name",
        "roll_number",
        "predicted_marks",
        "result",
        "date_time"
    ])
    history_df.to_csv(history_file, index=False)

# =====================================================
# SIDEBAR STYLE
# =====================================================
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827, #1f2933);
    color: white;
}
.profile-card {
    background-color: #1f2933;
    padding: 16px;
    border-radius: 14px;
    margin-bottom: 16px;
}
.profile-card h3 {
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOGIN / REGISTER
# =====================================================
if not st.session_state.logged_in:

    st.title("🔐 Teacher Login / Registration")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            user = teachers_df[
                (teachers_df["username"] == username) &
                (teachers_df["password"] == password)
            ]

            if not user.empty:
                st.session_state.logged_in = True
                st.session_state.teacher = user.iloc[0].to_dict()
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")

    with tab2:
        r_username = st.text_input("New Username")
        r_password = st.text_input("New Password", type="password")
        r_name = st.text_input("Full Name")
        r_subject = st.text_input("Subject")
        r_email = st.text_input("Email")

        if st.button("Register"):
            if r_username in teachers_df["username"].values:
                st.error("Username already exists")
            else:
                new_teacher = pd.DataFrame(
                    [[r_username, r_password, r_name, r_subject, r_email]],
                    columns=teachers_df.columns
                )
                teachers_df = pd.concat([teachers_df, new_teacher], ignore_index=True)
                teachers_df.to_csv(teachers_file, index=False)
                st.success("Registration successful! Please login.")

# =====================================================
# MAIN APP
# =====================================================
else:
    teacher = st.session_state.teacher
    username = teacher["username"]

    # ---------------- SIDEBAR PROFILE ----------------
    st.sidebar.markdown(f"""
    <div class="profile-card">
        <h3>👩‍🏫 Teacher Profile</h3>
        <p><b>Name:</b> {teacher['name']}</p>
        <p><b>Subject:</b> {teacher['subject']}</p>
        <p><b>Email:</b><br>{teacher['email']}</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.sidebar.radio(
        "📌 Navigation",
        ["Dashboard", "Predict Marks", "Prediction History"]
    )

    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.teacher = None
        st.rerun()

    # ---------------- LOAD DATASET ----------------
    data = pd.read_csv("student_performance.csv")

    X = data[['study_hours', 'attendance', 'previous_marks', 'assignments']]
    y_marks = data['final_marks']

    X_train, _, y_train, _ = train_test_split(
        X, y_marks, test_size=0.2, random_state=42
    )

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    data['pass_fail'] = data['final_marks'].apply(lambda x: 1 if x >= 50 else 0)
    log_model = LogisticRegression()
    log_model.fit(X, data['pass_fail'])

    # =====================================================
    # DASHBOARD
    # =====================================================
    if page == "Dashboard":
        st.title("📊 Teacher Dashboard")

        teacher_history = history_df[
            history_df["teacher_username"] == username
        ]

        col1, col2, col3 = st.columns(3)
        col1.metric("📈 Total Predictions", len(teacher_history))
        col2.metric("✅ Pass", len(teacher_history[teacher_history["result"] == "PASS"]))
        col3.metric("❌ Fail", len(teacher_history[teacher_history["result"] == "FAIL"]))

        if not teacher_history.empty:
            st.subheader("📉 Marks Trend")
            st.line_chart(teacher_history["predicted_marks"])
        else:
            st.info("No predictions yet.")

    # =====================================================
    # PREDICTION PAGE
    # =====================================================
    if page == "Predict Marks":
        st.title("🎯 Predict Student Result")

        student_name = st.text_input("Student Name")
        roll_number = st.text_input("Roll Number")

        study_hours = st.number_input("Study Hours / Day", 0.0, 15.0)
        attendance = st.number_input("Attendance (%)", 0.0, 100.0)
        previous_marks = st.number_input("Previous Marks", 0.0, 100.0)
        assignments = st.number_input("Assignments Completed", 0.0, 10.0)

        if st.button("Predict"):
            user_data = [[study_hours, attendance, previous_marks, assignments]]

            predicted_marks = lr_model.predict(user_data)[0]
            result = "PASS" if log_model.predict(user_data)[0] == 1 else "FAIL"

            st.success(f"📊 Predicted Marks: {predicted_marks:.2f}")
            st.success(f"🎓 Result: {result}")

            new_entry = pd.DataFrame([[ 
                username,
                student_name,
                roll_number,
                round(predicted_marks, 2),
                result,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ]], columns=history_df.columns)

            history_df = pd.concat([history_df, new_entry], ignore_index=True)
            history_df.to_csv(history_file, index=False)

    # =====================================================
    # HISTORY PAGE
    # =====================================================
    if page == "Prediction History":
        st.title("🧾 Prediction History")

        teacher_history = history_df[
            history_df["teacher_username"] == username
        ]

        if not teacher_history.empty:
            st.dataframe(
                teacher_history.drop(columns=["teacher_username"]),
                use_container_width=True
            )
        else:
            st.info("No records found.")
