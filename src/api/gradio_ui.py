import gradio as gr
import requests
import logging
from PIL import Image
from io import BytesIO
import base64

logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8000"

# ==============================================================================
# HEALTH PROBE CHECKER
# Pings the FastAPI liveness probe to sync UI state with Backend readiness.
# ==============================================================================
def check_server_status():
    """
    Silently pings the /health endpoint. Returns a formatted Markdown string
    with a color-coded status indicator.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            return "<div style='text-align: center; padding: 10px; background-color: #d4edda; color: #155724; border-radius: 5px;'><b>System Status: 🟢 ONLINE & READY (All 8 Evaluators Armed)</b></div>"
    except requests.exceptions.RequestException:
        pass # Server is down, booting up, or loading massive .pth files
        
    return "<div style='text-align: center; padding: 10px; background-color: #f8d7da; color: #721c24; border-radius: 5px;'><b>System Status: 🔴 OFFLINE / LOADING MODELS (Please wait...)</b></div>"

# ==============================================================================
# A/B EVALUATION LOGIC
# ==============================================================================
def predict_ab_test(image_a, image_b, prompt):
    """
    Handles the UI execution logic. Converts PIL numpy arrays to Base64 strings
    to match the FastAPI JSON schema, preventing Multipart Form parsing errors.
    """
    if image_a is None or image_b is None or not prompt:
        return "⚠️ Error: Please upload both images and enter a prompt.", None

    logger.info(f"[UI] Preparing Base64 JSON payload. Prompt: '{prompt}'")
    
    # Helper to convert Gradio numpy array -> PIL Image -> Base64 String
    def numpy_to_b64(img_array):
        pil_img = Image.fromarray(img_array)
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG", quality=95)
        # Encode to base64 and decode to string for JSON serialization
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Construct the exact JSON payload expected by the FastAPI backend
    payload = {
        "image_a_url": None,
        "image_a_b64": numpy_to_b64(image_a),
        "image_b_url": None,
        "image_b_b64": numpy_to_b64(image_b),
        "prompt": prompt
    }

    try:
        # Use json=payload instead of data= and files= to enforce Application/JSON
        response = requests.post(f"{API_BASE_URL}/evaluate/ab-test", json=payload)
        response.raise_for_status()
        result_data = response.json()
        
    except requests.exceptions.RequestException as e:
        logger.error(f"[UI] Connection failed: {e}")
        return f"❌ Connection Error: Backend refused the request. Details: {e}", None
    except Exception as e:
        logger.error(f"[UI] Unexpected error: {e}")
        return f"❌ Unexpected Error: {e}", None

    # Parse JSON for Bar Chart visualization
    evaluators_scores = {}
    detailed_explanations = "### 🤖 Detailed Evaluator Breakdown:\n\n"
    for eval_score in result_data['evaluator_scores']:
        name = eval_score['evaluator_name']
        
        # evaluators_scores[f"{name} (A)"] = eval_score['score_a']
        # evaluators_scores[f"{name} (B)"] = eval_score['score_b']
        
        purpose = eval_score.get('purpose', 'General evaluation')
        pref = eval_score.get('preferred', 'N/A')
        conf = eval_score.get('confidence', 0.0)
        
        evaluators_scores[f"{name} (Voted {pref})"] = conf
        
        # 3. Dynamically construct the Explanation sentence in the Frontend!
        expl_sentence = f"_{purpose}_ ➔ **Voted for Image {pref}** (Confidence: {conf * 100:.1f}%)"
        
        detailed_explanations += f"- **{name}**: {expl_sentence}\n"

    winner = result_data['overall_winner']
    
    summary_text = f"🏆 OVERALL WINNER: IMAGE {winner}\n"
    summary_text += f"**Executive Summary:** {result_data['reasoning_summary']}\n\n"
    summary_text += detailed_explanations

    return summary_text, evaluators_scores

# ==============================================================================
# UI LAYOUT DEFINITION
# ==============================================================================
with gr.Blocks(title="Vision-Reward-OS | A/B Testing") as demo:
    
    gr.Markdown("# 🤖 Vision-Reward-OS | Enterprise Image A/B Evaluator")
    
    status_indicator = gr.HTML(value="<div style='text-align: center; padding: 10px; background-color: #fff3cd; color: #856404; border-radius: 5px;'><b>System Status: 🟡 CHECKING CONNECTION...</b></div>")

    with gr.Row():
        with gr.Column():
            image_a = gr.Image(label="Image A (Candidate 1)", type="numpy")
        with gr.Column():
            image_b = gr.Image(label="Image B (Candidate 2)", type="numpy")
            
    prompt_input = gr.Textbox(label="Generation Prompt", placeholder="e.g., 'a cinematic shot of a cyberpunk city'")
    eval_btn = gr.Button("Evaluate Pair (Orchestrate 8 Evaluators)", variant="primary")

    with gr.Row():
        with gr.Column():
            output_text = gr.Markdown(label="Evaluation Summary")
        with gr.Column():
            output_chart = gr.Label(num_top_classes=8, label="Model Confidence Scores")

    eval_btn.click(
        fn=predict_ab_test, inputs=[image_a, image_b, prompt_input], outputs=[output_text, output_chart]
    )
    
    # --------------------------------------------------------------------------
    # BACKGROUND POLLING TRIGGER
    # Automatically pings the /health endpoint every 3 seconds in the background.
    # --------------------------------------------------------------------------
    ping_timer = gr.Timer(value=3)
    ping_timer.tick(fn=check_server_status, inputs=None, outputs=status_indicator)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8001, share=True)
