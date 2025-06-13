from flask import Flask, request, jsonify, render_template
from data import df
from utils import (
    rag, rag_image, comparison_evaluator,
    save_feedback_gpt2, save_feedback_blip, retrain_gpt2
)
import logging

app = Flask(__name__, template_folder='templates')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def home():
    return render_template('index(1).html')

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query_text = data.get('query')
    generated_answer, reference_answer = rag(query_text, df)
    lev_similarity, cos_sim, precision, recall, f1_score = comparison_evaluator(generated_answer, reference_answer)
    return jsonify({
        'generated_answer': generated_answer,
        'reference_answer': reference_answer,
        'lev_similarity': float(lev_similarity),
        'cosine_similarity': float(cos_sim),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score),
        'question': query_text
    })

@app.route('/caption', methods=['POST'])
def caption():
    image_file = request.files.get('image')
    question = request.form.get('question', '')
    if not image_file or not question:
        return jsonify({'error': 'Missing image or question'}), 400
    try:
        from PIL import Image
        image = Image.open(image_file.stream).convert('RGB')
        generated_answer, reference_answer = rag_image(image, question)
        # Convert reference_answer to string if it's a list
        if isinstance(reference_answer, list):
            reference_answer_str = " ".join(map(str, reference_answer))
        else:
            reference_answer_str = str(reference_answer)
        lev_similarity, cos_sim, precision, recall, f1_score = comparison_evaluator(generated_answer, reference_answer_str)
        return jsonify({
            'generated_caption': generated_answer,
            'reference_answer': reference_answer,
            'lev_similarity': float(lev_similarity),
            'cosine_similarity': float(cos_sim),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score)
        })
    except Exception as e:
        logger.error("Error in VQA inference: %s", e)
        return jsonify({'error': f'Failed to generate answer: {str(e)}'}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing JSON data'}), 400
    task_type = data.get('task_type', '').lower()
    user_input = data.get('user_input', '')
    corrected_output = data.get('corrected_output', '')
    if not task_type or not user_input or not corrected_output:
        return jsonify({'error': 'Missing required fields'}), 400
    try:
        if task_type == 'caption':
            save_feedback_blip(user_input, corrected_output)
        elif task_type == 'qa':
            save_feedback_gpt2(user_input, corrected_output)
        else:
            return jsonify({'error': 'Invalid task type'}), 400
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        return jsonify({'error': 'Failed to save feedback'}), 500
    return jsonify({'message': 'Feedback saved successfully'}), 200

from models import reload_gpt2_model

@app.route('/retrain', methods=['POST'])
def retrain():
    logger.info("Retraining the model...")
    try:
        retrain_gpt2()
        reload_gpt2_model()  # <-- Reload the model after retraining
        logger.info("Model retraining completed successfully.")
    except Exception as e:
        logger.error(f"Error during retraining: {e}")
        return jsonify({'error': 'Model retraining failed'}), 500
    return jsonify({'message': 'Model retrained and reloaded.'}), 200

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)