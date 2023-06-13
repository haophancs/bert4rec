from dotenv import load_dotenv

from ..src.reclib.helpers import Bert4RecPredictor

"""
DEMO FOR RECOMMENDATION TO USER WHO IS PROBABLY INTERESTED IN MARVEL CINEMATIC MOVIES
NOTE: INITIALIZING THE MODEL MAY TAKES MINUTES
"""

if __name__ == '__main__':
    load_dotenv()

    predictor = Bert4RecPredictor(

    )