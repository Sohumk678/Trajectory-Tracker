import streamlit as st
import ast, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ----- Constants -----
grade_map = {'A':4,'B':3,'C':2,'D':1,'E':0.5,'F':0}
# Update this path to where your training data is located locally
LABELED_CSV_PATH = "CS101_Student_Behavior.csv"

# ----- AI Model Helper Functions -----
def parse_list(x):
    if isinstance(x, list): return x
    if pd.isna(x): return []
    s = str(x)
    try:
        return ast.literal_eval(s)
    except Exception:
        try:
            return [ast.literal_eval(i) for i in s.split(',')]
        except Exception:
            return []

def numeric_grade(letter): return grade_map.get(str(letter), 0)
def list_numeric_grades(letters): return [numeric_grade(g) for g in letters]

def extract_assignment_metrics(assign_list):
    on_time, grades = [], []
    for a in assign_list:
        if isinstance(a, dict):
            on_time.append(1 if a.get('Submission_Status') == 'On Time' else 0)
            grades.append(numeric_grade(a.get('Grade','F')))
        else:  
            try:
                status, grade = a
                on_time.append(1 if status == 'On Time' else 0)
                grades.append(numeric_grade(grade))
            except Exception:
                pass
    return on_time, grades

def engineer_weekly_features(df, week_cutoff=8):
    att    = df['Weekly_Attendance'].apply(parse_list)
    logins = df['Weekly_Logins'].apply(parse_list)
    assigns= df['Assignments'].apply(parse_list)
    quizzes= df['Quizzes'].apply(parse_list)
    projs  = df['Programming_Projects'].apply(parse_list)

    quizzes_num = quizzes.apply(list_numeric_grades)
    projs_num   = projs.apply(list_numeric_grades)
    on_time_list, assign_grades_list = zip(*assigns.apply(extract_assignment_metrics))

    feats = pd.DataFrame({
        'Avg_Attendance':         att.apply(lambda x: np.mean(x[:week_cutoff]) if len(x)>0 else 0),
        'Avg_Logins':             logins.apply(lambda x: np.mean(x[:week_cutoff]) if len(x)>0 else 0),
        'Pct_OnTime_Assignments': pd.Series(on_time_list).apply(lambda x: np.mean(x[:week_cutoff]) if len(x)>0 else 0),
        'Avg_Quiz_Grade':         quizzes_num.apply(lambda x: np.mean(x[:week_cutoff]) if len(x)>0 else 0),
        'Avg_Project_Grade':      projs_num.apply(lambda x: np.mean(x) if len(x)>0 else 0),
    })

    def trend(vals):
        vals = vals[:week_cutoff]
        if len(vals) < 8: return 0.0
        return np.mean(vals[-4:]) - np.mean(vals[:4])

    feats['Trend_Attendance'] = att.apply(trend)
    feats['Trend_Logins']     = logins.apply(trend)
    feats['Trend_Quiz']       = quizzes_num.apply(trend)
    feats['Trend_OnTime']     = pd.Series(on_time_list).apply(trend)
    return feats

def standardise_columns(df):
    colmap = {}
    for col in df.columns:
        c = col.strip().lower()
        if c in ['student_id','id']:                                   colmap[col] = 'Student_ID'
        elif c.startswith('weekly_attendance'):                        colmap[col] = 'Weekly_Attendance'
        elif c.startswith('weekly_logins') or 'login' in c:            colmap[col] = 'Weekly_Logins'
        elif c.startswith('assignments'):                              colmap[col] = 'Assignments'
        elif c.startswith('quizzes') or c.startswith('weekly_quiz'):   colmap[col] = 'Quizzes'
        elif c.startswith('programming_projects') or 'projects' in c:  colmap[col] = 'Programming_Projects'
        elif c == 'gender':                                            colmap[col] = 'Gender'
        elif c == 'ethnicity':                                         colmap[col] = 'Ethnicity'
        elif c == 'scholarship':                                       colmap[col] = 'Scholarship'
        elif c == 'label':                                             colmap[col] = 'Label'
    return df.rename(columns=colmap)

def recommendation_from_label(label, prob_drift=None):
    if label == "Crisis":
        return ("Immediate intervention: schedule meeting, connect with advisor or counselor.",
                "Student is disengaged early. May need academic and emotional support.")
    elif label == "Drift":
        if prob_drift is not None and prob_drift >= 0.7:
            return ("Urgent mid-semester check-in; offer time management tips and study group options, consider academic coaching.",
                    "Performance started strong but shows significant decline. Prevent further drop.")
        else:
            return ("Mid-semester check-in; offer time management tips and study group options.",
                    "Performance started strong but declines. Prevent further drop.")
    else:
        return ("Congratulate on consistent performance.",
                "Stable performance. Encourage continued success or leadership roles.")

# ----- Model Training (Cached) -----
@st.cache_resource
def train_model():
    try:
        train_df = pd.read_csv(LABELED_CSV_PATH)
    except FileNotFoundError:
        st.error(f"Training data not found at {LABELED_CSV_PATH}. Please ensure the file exists.")
        return None, None

    train_df = standardise_columns(train_df)
    for col in ['Weekly_Attendance','Weekly_Logins','Assignments','Quizzes','Programming_Projects']:
        if col in train_df.columns:
            train_df[col] = train_df[col].apply(parse_list)

    X_full = engineer_weekly_features(train_df, week_cutoff=8)
    for c in ['Gender','Ethnicity','Scholarship']:
        X_full[c] = train_df[c].astype(str) if c in train_df.columns else 'Unknown'

    le = LabelEncoder()
    y  = le.fit_transform(train_df['Label'].astype(str))
    classes = list(le.classes_) 

    pre = ColumnTransformer(
        [('cat', OneHotEncoder(handle_unknown='ignore'), ['Gender','Ethnicity','Scholarship'])],
        remainder='passthrough'
    )
    base_rf = RandomForestClassifier(n_estimators=300, random_state=42)
    pipe_base = Pipeline([('pre', pre), ('clf', base_rf)])
    
    cal_rf = CalibratedClassifierCV(pipe_base, cv=3, method='sigmoid')
    cal_rf.fit(X_full, y)
    
    return cal_rf, classes

# ----- Prediction Logic -----
def score_dataframe(df, cal_rf, classes, week_cutoff=8):
    df = standardise_columns(df.copy())
    for col in ['Weekly_Attendance','Weekly_Logins','Assignments','Quizzes','Programming_Projects']:
        if col in df.columns:
            df[col] = df[col].apply(parse_list)

    feats = engineer_weekly_features(df, week_cutoff=week_cutoff)
    for c in ['Gender','Ethnicity','Scholarship']:
        feats[c] = df[c].astype(str) if c in df.columns else 'Unknown'

    proba = cal_rf.predict_proba(feats)
    class_to_idx = {c:i for i,c in enumerate(classes)}
    
    def p_of(label):
        return proba[:, class_to_idx[label]] if label in class_to_idx else np.zeros(len(df))

    pred_idx = np.argmax(proba, axis=1)
    pred_labels = [classes[i] for i in pred_idx]

    scored = df.copy()
    scored['Predicted_Label'] = pred_labels
    scored['Prob_Crisis'] = p_of('Crisis')
    scored['Prob_Drift']  = p_of('Drift')
    scored['Prob_Normal'] = p_of('Normal')

    recs = scored.apply(
        lambda r: recommendation_from_label(r['Predicted_Label'], prob_drift=r.get('Prob_Drift', None)),
        axis=1, result_type='expand'
    )
    scored['ML_Recommendation'] = recs[0]
    scored['ML_Instructor_Note'] = recs[1]
    return scored

# ----- Streamlit Pages -----
def page1():
    st.title("Trajectory Tracker: AI-Powered Recommendation System")
    st.write("Files must include the following columns: student_id, gender, ethnicity, scholarship, weekly_attendance, weekly_logins, assignments, quizzes, projects")
    
    uploaded_file = st.file_uploader("Choose a CSV file to analyze", type="csv")
    
    if uploaded_file is not None:
        st.success(f"Successfully uploaded: {uploaded_file.name}")
        df = pd.read_csv(uploaded_file)
        st.session_state['uploaded_df'] = df
        st.dataframe(df.head())

        missing_vals = df.isnull().sum()
        if missing_vals.any():
            st.warning("The amount of missing values in each column is as follows:")
            st.write(missing_vals.head())
    else:
        st.info("Please upload a CSV file to proceed.")

def page2():
    st.title("Current Data Analysis & Scoring")
    
    if 'uploaded_df' not in st.session_state:
        st.warning("Please upload a CSV file in the Home Page to see analytics.")
        return

    cal_rf, classes = train_model()
    if cal_rf is None:
        return

    st.write("### Interactive Rescoring")
    week = st.slider("Select Cutoff Week", min_value=4, max_value=12, value=8, step=1)
    
    with st.spinner("Scoring students..."):
        scored_df = score_dataframe(st.session_state['uploaded_df'], cal_rf, classes, week_cutoff=week)
        
    st.write(f"#### Preview of Scored Data (Week {week})")
    display_cols = ['Student_ID','Predicted_Label','Prob_Crisis','Prob_Drift','Prob_Normal','ML_Recommendation']
    st.dataframe(scored_df[[c for c in display_cols if c in scored_df.columns]].head(10))

    st.write("#### Risk Distribution")
    dist = scored_df['Predicted_Label'].value_counts()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    dist.plot(kind='bar', ax=ax, color=['#4C72B0', '#DD8452', '#55A868'])
    ax.set_title(f"Risk Distribution (Week {week})")
    ax.set_xlabel('Label')
    ax.set_ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)

def page3():
    st.title("Graphical Analysis of Student Data")
    st.write("Analysis and graphs of student data")
    
    if 'uploaded_df' not in st.session_state:
        st.warning("Please upload a CSV file in the Home Page to see analytics.")
        return
        
    student_ids = st.session_state['uploaded_df']['Student_ID'].unique()
    selected_student = st.selectbox("Select a Student ID to view their trajectory", student_ids)
    data_choice = st.selectbox("Select Data Type to Visualize", ['Weekly_Attendance', 'Weekly_Logins', 'Quizzes', 'Programming_Projects'])
    student_id_row = st.session_state['uploaded_df'][st.session_state['uploaded_df']['Student_ID'] == selected_student]
    attendance = parse_list(student_id_row["Weekly_Attendance"].values[0])
    logins = parse_list(student_id_row["Weekly_Logins"].values[0])
    quiz_grades = list_numeric_grades(parse_list(student_id_row["Quizzes"].values[0]))
    programming_projects = list_numeric_grades(parse_list(student_id_row["Programming_Projects"].values[0]))

    # Create the plotly graph based on the selected data type and the student id

    fig = go.Figure()
    if data_choice == 'Weekly_Attendance':
        fig.add_trace(go.Scatter(x = list(range(1, len(attendance)+1)), y=attendance, mode = 'lines+markers', name = 'Attendance'))
        fig.update_layout(title =f"Weekly Attendance for Student {selected_student}", xaxis_title = 'Week', yaxis_title = 'Attendance')
    elif data_choice == 'Weekly_Logins':
        fig.add_trace(go.Scatter(x = list(range(1, len(logins)+1)), y = logins, mode = 'lines+markers', name = 'Logins'))
        fig.update_layout(title =f"Weekly Logins for Student {selected_student}", xaxis_title = 'Week', yaxis_title = 'Logins')
    elif data_choice == 'Quizzes':
        fig.add_trace(go.Scatter(x=list(range(1, len(quiz_grades)+1)), y = quiz_grades, mode = 'lines+markers', name = 'Quizzes'))
        fig.update_layout(title =f"Weekly Quiz Grades for Student {selected_student}", xaxis_title = 'Week', yaxis_title = 'Quiz Grades')
    elif data_choice == 'Programming_Projects':
        fig.add_trace(go.Scatter(x=list(range(1, len(programming_projects)+1)), y = programming_projects, mode = 'lines+markers', name = 'Programming Projects'))
        fig.update_layout(title =f"Weekly Programming Projects for Student {selected_student}", xaxis_title = 'Week', yaxis_title = 'Programming Projects')

    st.plotly_chart(fig, use_container_width=True)


# ----- App Execution -----
if __name__ == "__main__":
    st.sidebar.success("Please select a page above.")
    pg = st.navigation([
        st.Page(page1, title="Home Page"), 
        st.Page(page2, title="Current Data Analysis"), 
        st.Page(page3, title="Graphical Analysis of Student Data")
    ])
    pg.run()