import logging
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
    tokenize,
)
from livekit.plugins import murf, silero, google, deepgram
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("sdr-agent")
load_dotenv(".env.local")


class SDRAgent(Agent):
    def __init__(self, company_data: dict) -> None:
        # Store company data for reference
        self.company_data = company_data
        self.lead_data = {
            "name": None,
            "company": None,
            "email": None,
            "role": None,
            "use_case": None,
            "team_size": None,
            "timeline": None,
        }
        self.conversation_history = []
        
        super().__init__(
            instructions=f"""You are a friendly Sales Development Representative (SDR) for {company_data['company_name']}.

COMPANY INFO:
- Name: {company_data['company_name']}
- Tagline: {company_data['tagline']}
- What we do: {company_data['description']}
- Target audience: {company_data['target_audience']}

YOUR PROCESS (FOLLOW THIS ORDER):
1. Greet warmly: "Hi! I'm [your name] from {company_data['company_name']}. Thanks for your interest!"
2. Ask discovery questions to understand their needs:
   - "What brings you here today?"
   - "What challenges are you currently facing with [relevant area]?"
   - "What made you interested in {company_data['company_name']}?"
3. Based on their responses, ask follow-up questions like:
   - "Tell me more about what you're working on"
   - "What's your current process for handling [their problem]?"
   - "What would an ideal solution look like for you?"
4. Answer any product/pricing questions using the answer_faq tool
5. Naturally collect required information during conversation:
   - name (ask early: "By the way, may I have your name?")
   - company (ask: "Which company are you with?" or "Where do you work?")
   - email (ask: "What's the best email to reach you at?")
   - role (ask: "What's your role there?" or "What do you do at [company]?")
   - use_case (should emerge from discovery questions - what they want to use product for)
   - team_size (ask: "How large is your team?" or "How many people would be using this?")
   - timeline (ask: "When are you looking to get started?" or "What's your timeline for this?")

QUALIFYING QUESTIONS TO ASK (Use these naturally in conversation):
- "What's your biggest pain point right now with [relevant topic]?"
- "Have you tried other solutions before? What didn't work?"
- "What's driving this need right now?"
- "Who else on your team would be involved in this decision?"
- "What's your budget range for something like this?"
- "How are you currently handling [problem] today?"
- "What would success look like for you in 3-6 months?"

CRITICAL RULES:
- Use update_lead_info tool IMMEDIATELY when user provides ANY information
- Ask questions ONE AT A TIME - don't overwhelm them
- Listen actively and acknowledge their answers before moving to next question
- BEFORE the user says goodbye, check what information you're still missing
- If missing info, say: "Before we wrap up, just a couple more quick details. Could I get your [missing field]?"
- DO NOT let the call end without collecting: name, email, company, role, use_case, team_size, and timeline
- Only use end_call_summary tool AFTER you have collected ALL required fields
- If they try to leave without giving info, politely insist: "I'd love to help, but I'll need your email so our team can follow up properly"

CONVERSATION STYLE:
- Keep it natural and conversational - you're speaking, not writing
- Be curious and consultative, not pushy
- Show genuine interest in their problems
- If they ask about pricing or features, use the answer_faq tool
- Don't use bullet points, emojis, or formatting in your speech
- Make it feel like a helpful conversation, not an interrogation

IMPORTANT: Check your progress regularly. If you've been talking for a while and still don't have their name, email, or other required fields, ASK FOR THEM.""",
        )
    
    @function_tool
    async def answer_faq(self, context: RunContext, user_question: str):
        """Search the company FAQ to answer user questions about the product, pricing, or company.
        
        Use this when the user asks questions like:
        - "What does your product do?"
        - "How much does it cost?"
        - "Do you have X feature?"
        - "Who is this for?"
        
        Args:
            user_question: The user's question that needs answering
            
        Returns:
            The best matching answer from the FAQ, or a message if no match found
        """
        logger.info(f"Searching FAQ for: {user_question}")
        
        # Simple keyword matching (you can make this more sophisticated)
        user_question_lower = user_question.lower()
        best_match = None
        best_score = 0
        
        for faq_item in self.company_data['faq']:
            question = faq_item['question'].lower()
            # Count matching words
            score = sum(1 for word in user_question_lower.split() if word in question)
            if score > best_score:
                best_score = score
                best_match = faq_item
        
        if best_match and best_score > 0:
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "user_question": user_question,
                "faq_matched": best_match['question'],
                "answer_given": best_match['answer']
            })
            return best_match['answer']
        else:
            # Fallback - return pricing info if question has price-related keywords
            if any(word in user_question_lower for word in ['price', 'cost', 'charge', 'fee', 'pricing']):
                pricing_info = json.dumps(self.company_data['pricing'], indent=2)
                return f"Here's our pricing structure: {pricing_info}"
            
            return "That's a great question. Let me check with our team and get back to you on that. In the meantime, is there anything else I can help you with?"
    
    @function_tool
    async def update_lead_info(
        self,
        context: RunContext,
        name: str = None,
        company: str = None,
        email: str = None,
        role: str = None,
        use_case: str = None,
        team_size: str = None,
        timeline: str = None
    ):
        """Store lead information as the conversation progresses.
        
        Call this IMMEDIATELY whenever the user provides any of these details.
        You don't need to collect all fields at once - call it whenever you get new info.
        
        REQUIRED FIELDS (must collect before ending call):
        - name: User's full name
        - company: Company they work for
        - email: Their email address
        - role: Their job title/role
        - use_case: What they want to use the product for
        - team_size: Size of their team (can be a number or range like "5-10 people")
        - timeline: When they want to start (e.g., "immediately", "next month", "Q2", "just exploring")
        
        Args:
            name: User's full name
            company: Company they work for
            email: Their email address
            role: Their job title/role
            use_case: What they want to use the product for
            team_size: Size of their team (number or range)
            timeline: When they're looking to get started
        """
        # Update only fields that are provided
        if name:
            self.lead_data['name'] = name
        if company:
            self.lead_data['company'] = company
        if email:
            self.lead_data['email'] = email
        if role:
            self.lead_data['role'] = role
        if use_case:
            self.lead_data['use_case'] = use_case
        if team_size:
            self.lead_data['team_size'] = team_size
        if timeline:
            self.lead_data['timeline'] = timeline
        
        # Calculate what's still missing
        missing_fields = [field for field, value in self.lead_data.items() if value is None]
        
        logger.info(f"Lead data updated: {self.lead_data}")
        logger.info(f"Missing fields: {missing_fields}")
        
        if missing_fields:
            return f"Information saved. Still need to collect: {', '.join(missing_fields)}"
        else:
            return "All required information collected!"
    
    @function_tool
    async def end_call_summary(self, context: RunContext):
        """Generate and save a summary when the call is ending.
        
        ONLY use this when:
        1. The user indicates they're done ("That's all", "Thanks bye",'No thanks' etc.)
        2. AND you have collected ALL required fields: name, company, email, role, use_case, team_size, timeline
        
        If any required fields are missing, DO NOT call this. Instead, ask the user for the missing information.
        
        Returns:
            A verbal summary to say to the user before ending
        """
        # Check if all required fields are collected
        missing_fields = [field for field, value in self.lead_data.items() if value is None]
        
        if missing_fields:
            logger.warning(f"Attempted to end call with missing fields: {missing_fields}")
            return f"Before we finish, I still need to get your {', '.join(missing_fields)}. Could you provide that for me?"
        
        timestamp = datetime.now().isoformat()
        
        # Create comprehensive summary
        summary = {
            "lead_id": f"LEAD_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": timestamp,
            "lead_info": self.lead_data,
            "conversation_history": self.conversation_history,
            "completeness": self._calculate_completeness()
        }
        
        # Save to JSON file
        leads_dir = Path("leads_data")
        leads_dir.mkdir(exist_ok=True)  # ✅ CREATE DIRECTORY IF IT DOESN'T EXIST
        
        leads_file = leads_dir / "leads.json"
        
        if leads_file.exists():
            with open(leads_file, "r") as f:
                try:
                    all_leads = json.load(f)
                except json.JSONDecodeError:
                    all_leads = []
        else:
            all_leads = []
        
        all_leads.append(summary)
        
        with open(leads_file, "w") as f:
            json.dump(all_leads, f, indent=2)
        
        logger.info(f"✓ Lead saved: {summary['lead_id']} at {leads_file.absolute()}")
        
        # Generate verbal summary
        name = self.lead_data.get('name')
        company = self.lead_data.get('company')
        use_case = self.lead_data.get('use_case')
        team_size = self.lead_data.get('team_size')
        timeline = self.lead_data.get('timeline')
        
        verbal_summary = f"Perfect! Thanks so much, {name}. Just to recap: you're from {company} with a team of {team_size}, interested in using our product for {use_case}, and looking to get started {timeline}. I've got all your details, and someone from our team will reach out to you shortly at the email you provided. Have a wonderful day!"
        
        return verbal_summary
    
    def _calculate_completeness(self):
        """Calculate what percentage of lead info was collected"""
        filled = sum(1 for v in self.lead_data.values() if v is not None)
        total = len(self.lead_data)
        return f"{filled}/{total} fields collected ({int(filled/total*100)}%)"


def load_company_data():
    """Load company data from JSON file"""
    data_file = Path("company_data.json")
    
    if not data_file.exists():
        # Return sample data if file doesn't exist
        logger.warning("company_data.json not found, using sample data")
        return {
            "company_name": "Zerodha",
            "tagline": "India's largest stockbroker",
            "description": "Online platform for trading stocks, mutual funds, and more with zero brokerage on delivery trades",
            "target_audience": "Individual traders and investors in India",
            "faq": [
                {
                    "question": "What does Zerodha do?",
                    "answer": "Zerodha is a discount broker offering trading in stocks, commodities, currencies, and mutual funds with zero brokerage on equity delivery trades."
                },
                {
                    "question": "What are the charges?",
                    "answer": "Account opening is free. Annual Maintenance Charge is Rs 300 plus GST per year. Zero brokerage on equity delivery and flat Rs 20 per trade for intraday and F&O."
                },
                {
                    "question": "Do you have a mobile app?",
                    "answer": "Yes, we have Kite mobile app for trading and Coin for mutual funds, both available on Android and iOS."
                },
                {
                    "question": "Who is this for?",
                    "answer": "This is perfect for individual traders and investors who want to trade stocks, F&O, or invest in mutual funds with minimal costs."
                }
            ],
            "pricing": {
                "account_opening": "Free",
                "amc": "Rs 300/year + GST",
                "equity_delivery": "Zero brokerage",
                "intraday_fno": "Rs 20 per trade"
            }
        }
    
    with open(data_file, "r") as f:
        return json.load(f)


def prewarm(proc: JobProcess):
    """Pre-load models to reduce latency"""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """Main entry point for the SDR agent"""
    
    ctx.log_context_fields = {"room": ctx.room.name}
    
    # Load company data
    company_data = load_company_data()
    logger.info(f"Loaded data for: {company_data['company_name']}")
    
    # Create voice pipeline
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",  
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )
    
    # Start session with SDR agent
    await session.start(
        agent=SDRAgent(company_data),
        room=ctx.room,
    )
    
    await ctx.connect()
    
    logger.info("📞 SDR Agent ready to capture leads!")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))