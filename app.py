
from ui import demo

if __name__ == "__main__":
    print("🚀 Launching BREAST AI PRO (Ultrasound) ...")
    print("🔬 Features: Segmentation + Classification, Uncertainty, Multi-tab Visualization")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        max_threads=10,
        ssl_verify=False,
        quiet=False,
        show_api=True
    )
