from google import genai
from app.config import settings
import PIL.Image
import io

class Gemini3Client:
    def __init__(self):
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(settings.GEMINI_3_MODEL)

    def analyze_frame(self, frame_data: bytes):
        """
        Analyzes a single frame using the Gemini 3.0 model.
        """
        image = PIL.Image.open(io.BytesIO(frame_data))
        prompt = """
        Analyze the image and provide the bounding box for the main object in the image.
        The output should be a JSON object with the following format:
        {
          "box_2d": [ymin, xmin, ymax, xmax]
        }
        """
        try:
            response = self.model.generate_content([prompt, image])
            return response.text
        except Exception as e:
            print(f"Error analyzing frame with Gemini 3.0: {e}")
            return None
