import gradio as gr
import requests
import logging
from PIL import Image
from io import BytesIO

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
    if image_a is None or image_b is None or not prompt:
        return "⚠️ Error: Please upload both images and enter a prompt.", None

    def pil_to_bytes(img):
        pil_img = Image.fromarray(img)
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG")
        return buffered.getvalue()

    files = [
        ('images', ('image_a.jpg', pil_to_bytes(image_a), 'image/jpeg')),
        ('images', ('image_b.jpg', pil_to_bytes(image_b), 'image/jpeg'))
    ]
    
    payload = {'prompt': prompt}

    try:
        response = requests.post(f"{API_BASE_URL}/evaluate/ab-test", files=files, data=payload)
        response.raise_for_status()
        result_data = response.json()
    except Exception as e:
        return f"❌ Connection Error: Ensure models have finished loading. Details: {e}", None

    evaluators_scores = {}
    for eval_score in result_data['scores']:
        name = eval_score['evaluator_name']
        evaluators_scores[f"{name} (A)"] = eval_score['score_a'] * 100
        evaluators_scores[f"{name} (B)"] = eval_score['score_b'] * 100

    winner = result_data['overall_winner']
    confidence = result_data['average_confidence'] * 100
    
    summary_text = f"🏆 OVERALL WINNER: IMAGE {winner}\n"
    summary_text += f"📊 Average Consensus Confidence: {confidence:.2f}%"

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
            output_text = gr.Textbox(label="Evaluation Summary", interactive=False)
        with gr.Column():
            output_chart = gr.Label(num_top_classes=16, label="Detailed Scores (Percentage Contribution)")

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
