✈️ AI-Powered Travel Planner
This project is an interactive AI-driven travel planner that generates personalized travel itineraries based on natural language input. Powered by OpenAI's GPT model, the planner understands travel preferences such as destination, duration, budget, interests, and group size to return a day-wise itinerary along with estimated costs and tailored suggestions.

![image](https://github.com/user-attachments/assets/e680f617-78f4-4bdc-985d-cf11bc388820)
![image](https://github.com/user-attachments/assets/2e42686e-92dd-48d3-afe4-5792435f81b3)
![image](https://github.com/user-attachments/assets/26bbae54-8c08-4e2f-8fb5-64eeb38f9abf)



🧠 Key Features
Natural Language Input: Users simply describe their dream trip (e.g., "luxury honeymoon in Santorini for 7 days, love food and sunsets, budget $6000").

AI-Powered Itinerary Generation: The OpenAI API interprets the request and generates detailed daily plans, complete with activity descriptions and costs.

Preference Extraction: Extracts and displays structured preferences such as destination, style, interests, group size, and budget.

AI Travel Insights: Provides smart summaries and tips based on the travel profile and selected destination.

Support for Multiple AI Modes: Choose between OpenAI, Hugging Face models, or a rule-based fallback.

📦 Setup & Usage
Install dependencies and activate your Python environment.

Run the app:

python travel.py
Choose your AI mode and optionally enter your API key.

Enter a sentence describing your dream vacation.

View your full itinerary and insights in the terminal.

📋 Example
Input:

Luxury honeymoon in Santorini for 7 days, love food and sunsets, budget $6000
Output includes:

7-day itinerary with restaurants, spa treatments, wine tastings, cruises

Budget estimates

Travel tips like “Perfect match for your luxury travel style”

🔧 Tech Stack
Python



OpenAI GPT API

Terminal-based UI



![image](https://github.com/user-attachments/assets/4e32621f-1084-41d1-9b87-67c304b6eea8)

![image](https://github.com/user-attachments/assets/23cc5e57-c107-462c-ae85-05c3d238cdab)

![image](https://github.com/user-attachments/assets/2249616d-2680-47b9-9dac-a4a20aa46956)


<img width="898" alt="Screenshot 2025-05-30 at 7 01 15 PM" src="https://github.com/user-attachments/assets/346312e4-8e2f-405f-874f-1d383647e807" />

<img width="933" alt="Screenshot 2025-05-30 at 7 01 45 PM" src="https://github.com/user-attachments/assets/af391f36-cb1c-40d3-b7c0-2b2e03450505" />

<img width="933" alt="Screenshot 2025-05-30 at 7 01 45 PM" src="https://github.com/user-attachments/assets/8cbce538-224c-4a5e-b8b7-f725e3eb6ba0" />


This AI-powered travel assistant uses a local Hugging Face model to generate personalized travel itineraries without requiring an OpenAI API key.

✅ Features (Local Mode – Option 2 and 3)
No internet/API key required: Runs entirely using my local machine.

NLP-based preference extraction: Parses user input for destination, budget, interests, travel style, and duration.

Generates day-by-day itinerary: Includes activities, dining spots, and experiences with costs and time slots.

AI Insights section: Explains how the suggested destination matches your stated preferences.

Fallback NLP engine: Uses rule-based or local model inference if transformer models are not installed.








