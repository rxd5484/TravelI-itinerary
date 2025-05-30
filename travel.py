# AI Travel Planner with LLM Integration
# Combines traditional NLP with modern LLM capabilities

import requests
import json
import re
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TravelPreferences:
    """User travel preferences extracted from natural language"""
    destination: Optional[str] = None
    duration_days: Optional[int] = None
    budget: Optional[float] = None
    interests: List[str] = None
    travel_style: Optional[str] = None
    season: Optional[str] = None
    group_size: int = 1
    special_requirements: List[str] = None

class OpenAILLMExtractor:
    """LLM-powered preference extraction using OpenAI"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.available = api_key is not None
        
        if self.available:
            try:
                import openai
                self.client = openai.OpenAI(api_key=api_key)
                logger.info("OpenAI LLM extractor initialized")
            except ImportError:
                logger.warning("OpenAI library not installed. Run: pip install openai")
                self.available = False
        else:
            logger.info("No OpenAI API key provided. Using fallback NLP.")
    
    def extract_preferences(self, user_input: str) -> TravelPreferences:
        """Extract preferences using LLM"""
        if not self.available:
            return self._fallback_extraction(user_input)
        
        try:
            prompt = f"""
            Analyze this travel request and extract structured information: "{user_input}"
            
            Return a JSON object with these exact fields:
            {{
                "destination": "specific city/country mentioned or null",
                "duration_days": "number of days as integer or null",
                "budget": "budget amount as number or null", 
                "interests": ["array of interest categories like culture, food, adventure, romance, etc."],
                "travel_style": "budget, moderate, or luxury",
                "group_size": "number of travelers as integer",
                "special_requirements": ["any specific needs mentioned"]
            }}
            
            Examples:
            - "romantic weekend in Paris" → {{"destination": "Paris", "duration_days": 2, "interests": ["romance"], "group_size": 2}}
            - "family trip for 5 days, budget $3000" → {{"duration_days": 5, "budget": 3000, "travel_style": "moderate", "group_size": 4}}
            
            Travel request: {user_input}
            
            Respond with ONLY the JSON object, no explanation.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            
            # Parse the JSON response
            content = response.choices[0].message.content.strip()
            if content.startswith('```json'):
                content = content.replace('```json', '').replace('```', '').strip()
            
            data = json.loads(content)
            
            # Convert to TravelPreferences object
            return TravelPreferences(
                destination=data.get('destination'),
                duration_days=data.get('duration_days'),
                budget=data.get('budget'),
                interests=data.get('interests', []),
                travel_style=data.get('travel_style', 'moderate'),
                group_size=data.get('group_size', 1),
                special_requirements=data.get('special_requirements', [])
            )
            
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return self._fallback_extraction(user_input)
    
    def _fallback_extraction(self, user_input: str) -> TravelPreferences:
        """Fallback to rule-based extraction if LLM fails"""
        prefs = TravelPreferences()
        user_input = user_input.lower()
        
        # Extract budget
        budget_match = re.search(r'\$(\d+(?:,\d{3})*)', user_input)
        if budget_match:
            prefs.budget = float(budget_match.group(1).replace(',', ''))
        
        # Extract duration
        duration_match = re.search(r'(\d+)\s*(?:days?|day)', user_input)
        if duration_match:
            prefs.duration_days = int(duration_match.group(1))
        
        # Extract interests
        interest_keywords = {
            'culture': ['culture', 'museum', 'history', 'art'],
            'food': ['food', 'cuisine', 'restaurant', 'culinary'],
            'romance': ['romantic', 'honeymoon', 'couple'],
            'adventure': ['adventure', 'hiking', 'extreme'],
            'beach': ['beach', 'ocean', 'sea', 'swimming'],
            'relaxation': ['relax', 'spa', 'peaceful']
        }
        
        interests = []
        for category, keywords in interest_keywords.items():
            if any(keyword in user_input for keyword in keywords):
                interests.append(category)
        prefs.interests = interests
        
        # Extract travel style
        if any(word in user_input for word in ['luxury', 'upscale', 'premium']):
            prefs.travel_style = 'luxury'
        elif any(word in user_input for word in ['budget', 'cheap', 'backpack']):
            prefs.travel_style = 'budget'
        else:
            prefs.travel_style = 'moderate'
        
        # Group size
        if any(word in user_input for word in ['honeymoon', 'romantic', 'couple']):
            prefs.group_size = 2
        
        return prefs

class HuggingFaceLLMExtractor:
    """Free local LLM using Hugging Face transformers"""
    
    def __init__(self):
        self.available = False
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            
            # Use a smaller, efficient model for text classification
            self.sentiment_analyzer = pipeline("sentiment-analysis")
            self.text_generator = pipeline("text-generation", 
                                          model="microsoft/DialoGPT-small",
                                          max_length=100)
            self.available = True
            logger.info("Hugging Face LLM extractor initialized")
            
        except ImportError:
            logger.warning("Transformers library not installed. Run: pip install transformers torch")
    
    def extract_preferences(self, user_input: str) -> TravelPreferences:
        """Extract preferences using local LLM"""
        if not self.available:
            return self._rule_based_extraction(user_input)
        
        try:
            # Use the model to understand sentiment and generate insights
            sentiment = self.sentiment_analyzer(user_input)[0]
            
            # Generate additional context
            prompt = f"Travel request: {user_input}. This suggests:"
            generated = self.text_generator(prompt, max_length=50, num_return_sequences=1)
            
            # Combine with rule-based extraction
            prefs = self._rule_based_extraction(user_input)
            
            # Enhance with LLM insights
            if sentiment['label'] == 'POSITIVE' and sentiment['score'] > 0.8:
                if 'luxury' not in prefs.travel_style:
                    prefs.travel_style = 'moderate'  # Upgrade style for very positive sentiment
            
            return prefs
            
        except Exception as e:
            logger.error(f"Local LLM extraction failed: {e}")
            return self._rule_based_extraction(user_input)
    
    def _rule_based_extraction(self, user_input: str) -> TravelPreferences:
        """Rule-based fallback"""
        # Same as fallback in OpenAI class
        return OpenAILLMExtractor()._fallback_extraction(user_input)

class AIItineraryGenerator:
    """LLM-powered itinerary generation"""
    
    def __init__(self, llm_extractor):
        self.llm = llm_extractor
        self.destinations = {
            "Paris": {"lat": 48.8566, "lon": 2.3522, "country": "France"},
            "Tokyo": {"lat": 35.6762, "lon": 139.6503, "country": "Japan"},
            "Bali": {"lat": -8.3405, "lon": 115.0920, "country": "Indonesia"},
            "Santorini": {"lat": 36.3932, "lon": 25.4615, "country": "Greece"},
            "Rome": {"lat": 41.9028, "lon": 12.4964, "country": "Italy"}
        }
    
    def generate_itinerary(self, user_input: str) -> Dict[str, Any]:
        """Generate complete itinerary using AI"""
        
        # Extract preferences using LLM
        preferences = self.llm.extract_preferences(user_input)
        
        # Select destination using AI reasoning
        destination = self._select_destination_ai(preferences, user_input)
        
        # Generate daily activities using LLM
        daily_itinerary = self._generate_daily_activities_ai(preferences, destination, user_input)
        
        # Calculate costs
        total_cost = self._calculate_costs(daily_itinerary, preferences)
        
        return {
            "destination": destination,
            "preferences": preferences,
            "daily_itinerary": daily_itinerary,
            "total_cost": total_cost,
            "ai_insights": self._generate_ai_insights(preferences, destination)
        }
    
    def _select_destination_ai(self, preferences: TravelPreferences, original_input: str) -> Dict[str, Any]:
        """Use AI to select the best destination"""
        
        if preferences.destination:
            # If user specified a destination, use it
            dest_name = preferences.destination
            if dest_name in self.destinations:
                dest_info = self.destinations[dest_name]
                return {
                    "name": dest_name,
                    "country": dest_info["country"],
                    "lat": dest_info["lat"],
                    "lon": dest_info["lon"]
                }
        
        # AI-powered destination selection
        if hasattr(self.llm, 'client') and self.llm.available:
            try:
                prompt = f"""
                Based on this travel request: "{original_input}"
                
                And these preferences:
                - Interests: {preferences.interests}
                - Budget: ${preferences.budget or 'not specified'}
                - Travel style: {preferences.travel_style}
                - Duration: {preferences.duration_days or 'not specified'} days
                - Group size: {preferences.group_size}
                
                Available destinations: {list(self.destinations.keys())}
                
                Which destination would be BEST and why? Respond with:
                destination_name|reason
                
                Example: Paris|Perfect for romance with world-class cuisine and art
                """
                
                response = self.llm.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=100
                )
                
                content = response.choices[0].message.content.strip()
                dest_name, reason = content.split('|', 1)
                dest_name = dest_name.strip()
                
                if dest_name in self.destinations:
                    dest_info = self.destinations[dest_name]
                    return {
                        "name": dest_name,
                        "country": dest_info["country"],
                        "lat": dest_info["lat"],
                        "lon": dest_info["lon"],
                        "ai_reason": reason.strip()
                    }
            except Exception as e:
                logger.error(f"AI destination selection failed: {e}")
        
        # Fallback to rule-based selection
        return self._fallback_destination_selection(preferences)
    
    def _fallback_destination_selection(self, preferences: TravelPreferences) -> Dict[str, Any]:
        """Rule-based destination selection"""
        # Simple logic for fallback
        if preferences.interests:
            if 'romance' in preferences.interests:
                dest_name = "Paris"
            elif 'beach' in preferences.interests:
                dest_name = "Bali"
            elif 'culture' in preferences.interests:
                dest_name = "Rome"
            else:
                dest_name = "Paris"  # Default
        else:
            dest_name = "Paris"  # Default
        
        dest_info = self.destinations[dest_name]
        return {
            "name": dest_name,
            "country": dest_info["country"],
            "lat": dest_info["lat"],
            "lon": dest_info["lon"]
        }
    
    def _generate_daily_activities_ai(self, preferences: TravelPreferences, 
                                    destination: Dict[str, Any], original_input: str) -> List[Dict]:
        """Generate daily activities using LLM"""
        
        duration = preferences.duration_days or 3
        
        if hasattr(self.llm, 'client') and self.llm.available:
            try:
                prompt = f"""
                Create a detailed {duration}-day luxury honeymoon itinerary for {destination['name']}, {destination['country']}.
                
                Requirements:
                - Budget: ${preferences.budget or 2000} total for 2 people
                - Focus on: romantic dining, sunset experiences, wine tastings
                - Style: Luxury with unique experiences each day
                - Include specific restaurant names and luxury activities
                
                Return ONLY valid JSON in this exact format:
                [
                  {{
                    "day": 1,
                    "date": "2025-06-01", 
                    "activities": [
                      {{"time": "09:00", "activity": "Private Catamaran Cruise", "cost": 180, "description": "Exclusive sailing around the caldera with breakfast"}},
                      {{"time": "13:00", "activity": "Lunch at Kastro Restaurant", "cost": 85, "description": "Fine dining with caldera views"}},
                      {{"time": "16:00", "activity": "Couples Spa at Mystique Hotel", "cost": 220, "description": "Luxury spa treatment with sunset views"}},
                      {{"time": "20:00", "activity": "Dinner at La Maison Oia", "cost": 150, "description": "Michelin-level cuisine with sunset terrace"}}
                    ]
                  }}
                ]
                
                Make each day unique with different luxury experiences. Include specific venue names.
                """
                
                response = self.llm.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4,
                    max_tokens=1500
                )
                
                content = response.choices[0].message.content.strip()
                if content.startswith('```json'):
                    content = content.replace('```json', '').replace('```', '').strip()
                
                return json.loads(content)
                
            except Exception as e:
                logger.error(f"AI itinerary generation failed: {e}")
        
        # Fallback to simple itinerary
        return self._generate_simple_itinerary(duration, destination, preferences)
    
    def _generate_simple_itinerary(self, duration: int, destination: Dict[str, Any], 
                                 preferences: TravelPreferences) -> List[Dict]:
        """Simple fallback itinerary generation"""
        itinerary = []
        
        # Basic activities by destination
        activities = {
            "Paris": ["Louvre Museum", "Eiffel Tower", "Seine Cruise", "Montmartre", "Versailles"],
            "Tokyo": ["Senso-ji Temple", "Shibuya Crossing", "Tsukiji Market", "Imperial Palace", "Harajuku"],
            "Bali": ["Uluwatu Temple", "Rice Terraces", "Beach Day", "Monkey Forest", "Volcano Hike"],
            "Rome": ["Colosseum", "Vatican City", "Trevi Fountain", "Pantheon", "Roman Forum"],
            "Santorini": ["Oia Sunset", "Wine Tasting", "Fira Walk", "Red Beach", "Akrotiri"]
        }
        
        dest_activities = activities.get(destination["name"], activities["Paris"])
        
        for day in range(1, duration + 1):
            activity_idx = (day - 1) % len(dest_activities)
            main_activity = dest_activities[activity_idx]
            
            cost = 50 if preferences.travel_style == 'luxury' else 25
            
            day_plan = {
                "day": day,
                "date": (datetime.now() + timedelta(days=day-1)).strftime("%Y-%m-%d"),
                "activities": [
                    {"time": "09:00", "activity": main_activity, "cost": cost, 
                     "description": f"Explore {main_activity}"},
                    {"time": "13:00", "activity": "Local Lunch", "cost": cost//2, 
                     "description": "Authentic local cuisine"},
                    {"time": "15:00", "activity": "Free Exploration", "cost": 0, 
                     "description": "Discover the neighborhood"},
                    {"time": "19:00", "activity": "Dinner", "cost": cost*1.5, 
                     "description": "Evening dining experience"}
                ]
            }
            itinerary.append(day_plan)
        
        return itinerary
    
    def _calculate_costs(self, itinerary: List[Dict], preferences: TravelPreferences) -> Dict[str, float]:
        """Calculate total costs"""
        activity_cost = 0
        for day in itinerary:
            for activity in day["activities"]:
                activity_cost += activity.get("cost", 0)
        
        group_size = preferences.group_size
        duration = len(itinerary)
        
        # Estimate other costs
        accommodation = 100 * duration if preferences.travel_style == 'luxury' else 50 * duration
        transportation = 200 if preferences.travel_style == 'luxury' else 100
        
        total_per_person = activity_cost + accommodation + transportation
        total_for_group = total_per_person * group_size
        
        return {
            "activities": activity_cost * group_size,
            "accommodation": accommodation,
            "transportation": transportation * group_size,
            "total_per_person": total_per_person,
            "total_for_group": total_for_group
        }
    
    def _generate_ai_insights(self, preferences: TravelPreferences, 
                            destination: Dict[str, Any]) -> List[str]:
        """Generate AI-powered travel insights"""
        insights = [
            f"🎯 Perfect match for your {preferences.travel_style} travel style",
            f"💡 {destination['name']} aligns well with your interests: {', '.join(preferences.interests)}",
            f"👥 Great for groups of {preferences.group_size}"
        ]
        
        if hasattr(self.llm, 'client') and self.llm.available:
            try:
                prompt = f"""
                Generate 3 unique travel insights for someone going to {destination['name']} with these preferences:
                - Interests: {preferences.interests}
                - Travel style: {preferences.travel_style}
                - Budget: ${preferences.budget or 'flexible'}
                
                Each insight should be practical and specific. Format as simple bullet points.
                """
                
                response = self.llm.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6,
                    max_tokens=200
                )
                
                ai_insights = response.choices[0].message.content.strip().split('\n')
                insights.extend([insight.strip('- •') for insight in ai_insights if insight.strip()])
                
            except Exception as e:
                logger.error(f"AI insights generation failed: {e}")
        
        return insights[:5]  # Limit to 5 insights

class EnhancedTravelPlannerApp:
    """Main application with LLM integration"""
    
    def __init__(self, openai_api_key: str = None, use_local_llm: bool = False):
        if openai_api_key:
            self.llm_extractor = OpenAILLMExtractor(openai_api_key)
            logger.info("Using OpenAI LLM for enhanced AI features")
        elif use_local_llm:
            self.llm_extractor = HuggingFaceLLMExtractor()
            logger.info("Using local Hugging Face LLM")
        else:
            self.llm_extractor = OpenAILLMExtractor()  # Will fall back to rule-based
            logger.info("Using rule-based NLP (no LLM)")
        
        self.itinerary_generator = AIItineraryGenerator(self.llm_extractor)
    
    def plan_trip(self, user_input: str) -> Dict[str, Any]:
        """Plan a trip using AI"""
        logger.info(f"🤖 AI processing travel request: {user_input}")
        
        result = self.itinerary_generator.generate_itinerary(user_input)
        
        return result
    
    def format_output(self, result: Dict[str, Any]) -> str:
        """Format the AI-generated itinerary"""
        destination = result["destination"]
        preferences = result["preferences"]
        costs = result["total_cost"]
        
        output = []
        output.append(f"🤖 AI TRAVEL PLANNER RESULTS")
        output.append(f"🌟 DESTINATION: {destination['name'].upper()}, {destination['country'].upper()}")
        
        if destination.get("ai_reason"):
            output.append(f"🎯 AI Recommendation: {destination['ai_reason']}")
        
        output.append(f"\n📊 EXTRACTED PREFERENCES:")
        output.append(f"• Duration: {preferences.duration_days or 'flexible'} days")
        output.append(f"• Budget: ${preferences.budget or 'flexible'}")
        output.append(f"• Interests: {', '.join(preferences.interests) if preferences.interests else 'general'}")
        output.append(f"• Travel Style: {preferences.travel_style}")
        output.append(f"• Group Size: {preferences.group_size}")
        
        output.append(f"\n💰 TOTAL COST: ${costs['total_for_group']:.2f} for {preferences.group_size} people")
        
        output.append(f"\n📅 AI-GENERATED ITINERARY:")
        for day in result["daily_itinerary"]:
            output.append(f"\n--- Day {day['day']} ({day['date']}) ---")
            for activity in day["activities"]:
                cost_str = f"(${activity['cost']})" if activity['cost'] > 0 else "(Free)"
                output.append(f"{activity['time']} - {activity['activity']} {cost_str}")
                output.append(f"   {activity['description']}")
        
        output.append(f"\n🧠 AI INSIGHTS:")
        for insight in result["ai_insights"]:
            output.append(f"• {insight}")
        
        return "\n".join(output)

# Example usage
if __name__ == "__main__":
    print("🤖 AI-Powered Travel Planner")
    print("Choose your AI mode:")
    print("1. OpenAI GPT (requires API key)")
    print("2. Local Hugging Face LLM (free)")
    print("3. Rule-based NLP (no AI)")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        api_key = input("Enter OpenAI API key (or press Enter to skip): ").strip()
        app = EnhancedTravelPlannerApp(openai_api_key=api_key if api_key else None)
    elif choice == "2":
        app = EnhancedTravelPlannerApp(use_local_llm=True)
    else:
        app = EnhancedTravelPlannerApp()
    
    print("\n🌟 AI Travel Planner Ready!")
    print("Try: 'luxury honeymoon in Santorini for 7 days, love food and sunsets, budget $6000'")
    
    while True:
        user_input = input("\n✈️ Describe your dream trip (or 'quit'): ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("🌍 Happy travels!")
            break
        
        if user_input.strip():
            print("\n🤖 AI is planning your trip...\n")
            result = app.plan_trip(user_input)
            formatted_output = app.format_output(result)
            print(formatted_output)
            print("\n" + "=" * 80)
        else:
            print("Please describe your travel preferences!")