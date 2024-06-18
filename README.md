
# **Resume Reboot: Your ATS Optimization Expert**

Welcome to Resume Reboot, an advanced AI-powered application designed to enhance your job application process by providing detailed feedback on your resume. This tool leverages cutting-edge AI technology to help you optimize your resume for better matches with job descriptions and improve your chances of landing your dream job.

## **Project Overview**

The job application process can be daunting, especially when you receive rejections without any feedback. This project aims to address this issue by offering an intuitive tool for job seekers to analyze and improve their resumes using Google Generative AI (GEMINI).

## **Objectives**

- **Intuitive Tool:** Provide an easy-to-use tool for job seekers to match their resumes with job descriptions.
- **Advanced AI:** Leverage state-of-the-art AI technology to analyze and provide feedback on resumes.
- **User-Friendly Interface:** Offer a streamlined and user-friendly interface to simplify the resume review process.

## **Features**

1. **Resume Upload:**
   - Upload your resume in PDF format.

2. **Job Description Input:**
   - Paste the job description of the position you are targeting.

3. **AI-Powered Analysis:**
   - Utilize GEMINI AI to provide a detailed analysis of your resume in context with the job description.

4. **Feedback on Different Aspects:**
   - **Resume Review:** General feedback on your resume.
   - **Skills Improvement:** Suggestions for enhancing your skills.
   - **Keywords Analysis:** Identification of missing keywords in your resume.
   - **Match Percentage:** A percentage score indicating how well your resume matches the job description.

## **Technologies Used**

- **Streamlit:** For creating the web application interface.
- **Google Generative AI (Gemini Pro Vision):** For processing and analyzing resume content.
- **Python:** The primary programming language used for backend development.
- **PDF2Image & PIL:** For handling PDF file conversions and image processing.

## **Challenges Faced**

- **Integration with GEMINI AI:**
  - Ensuring seamless communication between the Streamlit interface and the GEMINI AI model.
  
- **PDF Handling:**
  - Efficiently converting PDF content to a format suitable for analysis by the AI model.
  
- **User Experience Optimization:**
  - Creating an intuitive and responsive user interface.

## **Future Enhancements**

- **Support for Multiple Pages:**
  - Extend functionality to handle multi-page resumes.
  
- **Customizable Feedback Categories:**
  - Allow users to choose specific areas for feedback.
  
- **Interactive Resume Editing:**
  - Integrate a feature to edit the resume directly based on AI suggestions.
  
- **Enhanced Error Handling:**
  - Improve the system's robustness in handling various file formats and user inputs.

## **Conclusion**

Resume Reboot stands as a significant tool in bridging the gap between job seekers and their ideal job roles. By harnessing the power of AI, it provides valuable insights and recommendations, making it a pivotal step in enhancing the job application process.

## **Getting Started**

### Prerequisites

- Python 3.x
- Streamlit
- PyPDF2
- Google Generative AI (GEMINI) API key
- Other dependencies as listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/resume-reboot.git
   cd resume-reboot
   ```

2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   - Create a `.env` file in the root directory and add your Google API key:
     ```env
     GOOGLE_API_KEY=your_google_api_key
     ```

### Running the Application

1. Start the Streamlit app:
   ```sh
   streamlit run app.py
   ```

2. Open your browser and go to `http://localhost:8501`.

### Usage

1. Upload your resume in PDF format.
2. Paste the job description of the position you are applying for.
3. Click on the appropriate buttons to receive feedback, skills recommendations, keyword analysis, and match percentage.

## **Contributing**

We welcome contributions to improve Resume Reboot. Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Create a pull request.

## **License**

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

By following this README, users can easily understand, set up, and use the Resume Reboot application. Feel free to modify the details as needed.
