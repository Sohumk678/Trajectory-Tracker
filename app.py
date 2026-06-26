import streamlit as st
import pandas as pd
import plotly.express as px

uploaded_file = None


st.sidebar.success("Please select a page above.")

# reads the csv file and displays it in the app
def read_csv(file):
    df = pd.read_csv(file)
    return df

# home page where files can be uploaded and displayed
# this can also be turned into solely a title page that informs the user about the app and its purpose
def page1():
    st.title("Trajectory Tracker: AI-Powered Reccommendation System")

    # global for use in other pages
    global uploaded_file
    st.write("Files must include the following columns: student_id, gender, ethnicity, scholarship, weekly_attendance, weekly_logins, assignments, quizzes, projects")
    # can be changed to accept multiple file types or multiple files at once
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        st.write("You Uploaded " + uploaded_file.name)
        df = read_csv(uploaded_file)
        st.write(df)
        
        

        # Code probably not needed since the app will just throw an error if the columns are missing and an error message will be displayed
        # required_columns = ["student_id", "gender", "ethnicity", "scholarship", "weekly_attendance", "weekly_logins", "assignments", "quizzes", "projects"] 
        # for header in required_columns:
            # if header not in df.columns:
                # st.error(f"Column '{header}' is missing from the uploaded file.")
    
    else:
        st.write("Please upload a CSV file to proceed.")
    
    
    

# page for current data trends up to the cutoff week including graphs and analysis
def page2():
    st.title("Current Data Analysis")
    st.write("Analysis and graphs of the current data uploaded in page 1.")
    # makes sure users know why the page is empty if they haven't uploaded a file yet
    if uploaded_file is None:
        st.write("Please upload a CSV file in the Home Page to see analytics.")

# page for future data trends and predictions based on the current data
def page3():
    st.title("Future Data Analysis and Predictions")
    st.write("Analysis and graphs of the trajectory of student data")
    # makes sure users know why the page is empty if they haven't uploaded a file yet
    if uploaded_file is None:
        st.write("Please upload a CSV file in the Home Page to see analytics.")

# can also add other customizations like an icon
pg = st.navigation([st.Page(page1, title="Home Page"), st.Page(page2, title="Current Data Analysis"), st.Page(page3, title="Future Data Analysis and Predictions")])


pg.run()








        

    