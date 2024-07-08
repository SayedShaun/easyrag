from rag.easyrag import GoogleGemini

model = GoogleGemini(pdf_path="Letter of Recommendation.pdf", google_api_key="AIzaSyAM5le2LaCfVY9EjAql3K__F_k0U6cm5AE")
model.retrieve_answer(continuous_chat=True)