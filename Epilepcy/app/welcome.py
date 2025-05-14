import streamlit as st

def welcome():
    # Custom background color
    st.markdown(
        """
        <style>
            body {
                background-color: #DAF7A6;
            }
            .fade-in {
                animation: fadeIn 2s ease-in-out;
            }
            @keyframes fadeIn {
                0% { opacity: 0; }
                100% { opacity: 1; }
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🧠 Epilepsy Detection System")
    st.markdown("Welcome to the ML-powered epilepsy detection platform.")

    # Centered image with fade-in animation
    image_url = "https://www.medicalindependent.ie/wp-content/uploads/2020/10/brain.jpg"

    st.markdown(
        f"""
        <div class="fade-in" style="text-align: center; margin-top: 20px;">
            <img src="{image_url}" alt="Brain EEG" style="width:70%; max-width:600px; border-radius: 10px;">
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("This application uses machine learning algorithms to analyze EEG data and predict the likelihood of epilepsy.")
    st.markdown("The system is designed to assist healthcare professionals in making informed decisions.")
    st.markdown("Please proceed to the login page to access the system.")

    st.markdown("---")
