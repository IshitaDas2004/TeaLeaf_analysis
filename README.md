# TeaLeaf_analysis
TeaDiagnosticX --------- AI-powered tea plant disease detection and diagnosis web application.
Overview:-
TeaDiagnosticX is a deep learning-based web application that allows farmers, researchers, and tea garden managers to upload a photo of a tea leaf and instantly receive a disease diagnosis along with recommended treatment. It is built using a Convolutional Neural Network (CNN) trained on real tea leaf images.
Features:-
📸 Image Upload — Upload a tea leaf photo directly from your device
🔬 AI Diagnosis — CNN model classifies the leaf condition with a confidence score
💊 Treatment Recommendations — Disease-specific fungicide and care guidance
👤 User Authentication — Register and log in to save your analysis history
🗃️ History Tracking — All predictions saved to your account in a database
📱 Responsive UI — Works on desktop and mobile browsers
🍃Detectable Diseases
Disease Description:- 
Healthy Leaf          No disease detected
Gray Blight           A fungal infection causing gray patches
Brown Blight          Browning and necrosis of leaf tissue
Red Rust              Algal disease, causing reddish deposits
Leaf Spot             Circular spots caused by fungal pathogens
🛠️Tech Stack:-
Layer               Technology
Frontend            HTML5, CSS3, Vanilla JavaScript
Backend             Python, Flask, Flask-CORSDatabaseSQLite via Flask-SQLAlchemy
ML Framework        TensorFlow / Keras
Image Processing    Pillow (PIL)
Auth                Werkzeug password hashing 
Model Storage       Git LFS


Project Structure:-
Backend/
│
├── app.py                    # Flask application & routes
├── Train.py                  # Model training script
├── Download_Dataset.py       # Dataset download utility
├── class_names.txt           # Disease class labels
├── tea_leaf_model.keras      # Trained CNN model (Git LFS)
├── requirements.txt          # Python dependencies
├── .gitignore
│
├── templates/
│   ├── home.html             # Landing page
│   ├── analyze.html          # Leaf upload & results page
│   ├── login.html            # User login page
│   └── register.html         # User registration page
│
└── static/
    └── style.css             # Global stylesheet


🗄️ Database
The app uses SQLite and creates the database automatically on first run.
Tables:
  1. users — Stores registered user details (name, email, phone, location, hashed password)
  2. predictions — Stores uploaded images, prediction results, confidence scores, and treatment info linked to each user

The database file teadiagnosticx.db is excluded from the repository via .gitignore.


🧠 Model Details
Property                    Value
Architecture                Deep Residual CNN
Input Size                  128 × 128 px
Output Classes              5
Format                      .keras
Size                        93.6 MB (stored via Git LFS)


📸 How to Use
    1. Open the app in your browser
    2. Register for a free account or log in
    3. Click Analyze Leaf
    4. Upload a clear, close-up photo of a single tea leaf
    5. Click Run Diagnosis
    6. View the predicted disease, confidence score, and recommended treatment


🔒 Security
  1. Passwords are hashed using Werkzeug's generate_password_hash
  2. Sessions are managed securely via Flask's secret key
  3. Raw image bytes are stored in the database as BLOBs linked to user accounts.


👩‍💻 Author
Ishita Das
