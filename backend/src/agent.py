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

logger = logging.getLogger("dnd-game-master")
load_dotenv(".env.local")


class GameMasterAgent(Agent):
    def __init__(self, universe: str = "detective", tone: str = "dramatic") -> None:
        self.universe = universe
        self.tone = tone
        self.turn_count = 0
        self.story_events = []
        self.player_name = None
        self.current_location = None
        self.inventory = []
        self.companions = []
        self.clues = []  # For detective universe
        self.suspects = []  # For detective universe
        
        # Define the universe settings
        universe_settings = {
            "fantasy": {
                "setting": "a medieval fantasy realm of magic, dragons, and ancient kingdoms",
                "starting_location": "the bustling market square of Thornhaven, a frontier town",
                "threats": "bandits, monsters, dark wizards, and ancient curses",
                "tone_desc": "epic and adventurous"
            },
            "sci-fi": {
                "setting": "a distant future among the stars, where humanity has colonized multiple planets",
                "starting_location": "the cargo bay of the starship Odyssey, docked at Station Epsilon",
                "threats": "alien creatures, rogue AI, space pirates, and corporate conspiracies",
                "tone_desc": "mysterious and tense"
            },
            "post-apocalypse": {
                "setting": "a world devastated by nuclear war, where survivors struggle in the wasteland",
                "starting_location": "the ruins of what was once a shopping mall, now a survivor settlement",
                "threats": "raiders, mutants, radiation storms, and scarce resources",
                "tone_desc": "gritty and survival-focused"
            },
            "horror": {
                "setting": "a small town plagued by supernatural forces and dark secrets",
                "starting_location": "an old Victorian mansion on the outskirts of Ravencrest",
                "threats": "ghosts, demons, cultists, and eldritch horrors",
                "tone_desc": "spooky and atmospheric"
            },
            "detective": {
                "setting": "a noir-style city in the 1940s, where crime and corruption run deep",
                "starting_location": "your cramped detective office on the third floor of a rundown building on 5th Street",
                "threats": "murderers, crime syndicates, corrupt officials, and dark conspiracies",
                "tone_desc": "noir and mysterious, with sharp observations and clever deductions"
            }
        }
        
        self.world = universe_settings.get(universe, universe_settings["detective"])
        
        super().__init__(
            instructions=f"""You are an expert Game Master (GM) running a {tone} {universe} tabletop RPG adventure.

UNIVERSE: {self.world['setting']}

YOUR ROLE AS GM:
1. You describe scenes vividly and immersively
2. You control NPCs (non-player characters) and the environment
3. You react to the player's choices and drive the story forward
4. You maintain continuity - remember what happened before
5. You create challenges, mysteries, and opportunities for the player

STORYTELLING RULES:
- Use {self.world['tone_desc']} language appropriate to the {tone} tone
- Keep descriptions extremely concise (1-2 sentences) before prompting player action
- NO long descriptions - get to the action immediately
- React logically to player choices - if they try something clever, it might work
- If they try something impossible, quickly explain why and offer an alternative
- Track important details: locations visited, NPCs met, items found
- Build toward a quick climax by turn 6
- Actively push story toward conclusion after turn 5

CONVERSATION FLOW:
1. Describe the current scene or situation
2. Present choices or challenges (not always explicit - let them be creative)
3. ALWAYS end with a question prompting action:
   - "What do you do?"
   - "How do you respond?"
   - "What's your next move?"
4. When player responds, describe the outcome and consequences
5. Continue the story based on their action

IMPORTANT CONTINUITY:
- Reference past events: "You remember the merchant mentioned this place..."
- Track relationships: "The guard recognizes you from earlier..."
- Build consequences: "Your earlier choice to spare the bandit has consequences..."
- Remember items/abilities: "You still have the rusty key from the tavern..."

DETECTIVE UNIVERSE SPECIAL RULES (if applicable):
- Use record_clue tool when player discovers evidence
- Use add_suspect tool when player encounters persons of interest
- Use review_case_notes when player wants to review their investigation
- Present red herrings and misleading evidence
- Allow player to interrogate suspects and examine crime scenes
- Build toward revealing the culprit based on accumulated evidence
- Use noir language: "dame", "gumshoe", "case", "lead", "stiff" (for corpse)

PACING (CRITICAL - KEEP STORIES SHORT):
- Target: Complete story in 6-8 conversational turns (approximately 5-7 minutes)
- Turn 1: Quick intro and immediate hook
- Turns 2-4: Rapid escalation with 1-2 challenges
- Turns 5-6: Quick climax
- Turn 7: Fast resolution and ending
- Keep each response to 2-3 sentences maximum
- Move the story forward quickly - don't linger on descriptions
- After turn 6, start wrapping up the story
- Use phrases like "Time is running out" or "This is it" to signal climax

DO NOT:
- Make choices for the player
- Tell them what their character thinks or feels
- Railroad them into one solution
- Forget previous events or conversations
- Use meta-gaming language (don't say "roll for", "make a check", etc.)

START: Begin by asking the player their character's name, then launch into the opening scene at {self.world['starting_location']}.""",
        )
    
    @function_tool
    async def record_event(
        self, 
        context: RunContext, 
        event_type: str, 
        description: str,
        location: str = None
    ):
        """Record a significant story event for continuity tracking.
        
        Use this tool to remember important story beats so you can reference them later.
        
        Args:
            event_type: Type of event - "combat", "discovery", "npc_interaction", "location_change", "item_acquired"
            description: Brief description of what happened
            location: Where it happened (optional)
            
        Returns:
            Confirmation that event was recorded, plus pacing guidance
        """
        self.turn_count += 1
        
        event = {
            "turn": self.turn_count,
            "type": event_type,
            "description": description,
            "location": location or self.current_location,
            "timestamp": datetime.now().isoformat()
        }
        
        self.story_events.append(event)
        
        # Update current location if it changed
        if event_type == "location_change" and location:
            self.current_location = location
        
        logger.info(f"📖 Story event recorded (Turn {self.turn_count}): {event_type} - {description}")
        
        # Provide pacing guidance
        pacing_msg = f"Event recorded: {description}. "
        
        if self.turn_count >= 7:
            pacing_msg += "WRAP UP NOW - End the story in the next response with a satisfying conclusion."
        elif self.turn_count >= 5:
            pacing_msg += "CLIMAX - Build to the final confrontation or revelation NOW."
        elif self.turn_count >= 3:
            pacing_msg += "ESCALATE - Present the main challenge soon."
        
        return pacing_msg
    
    @function_tool
    async def update_inventory(
        self, 
        context: RunContext, 
        item: str,
        action: str = "add"
    ):
        """Track items the player acquires or loses.
        
        Use this when player finds, buys, or loses items.
        
        Args:
            item: Name of the item
            action: "add" to give item to player, "remove" to take it away
            
        Returns:
            Confirmation message
        """
        if action == "add":
            self.inventory.append(item)
            logger.info(f"🎒 Added to inventory: {item}")
            return f"Added {item} to inventory"
        elif action == "remove":
            if item in self.inventory:
                self.inventory.remove(item)
                logger.info(f"🎒 Removed from inventory: {item}")
                return f"Removed {item} from inventory"
            else:
                return f"Player doesn't have {item}"
        
        return "Invalid action"
    
    @function_tool
    async def add_companion(self, context: RunContext, npc_name: str, description: str):
        """Track NPCs who join the player as companions.
        
        Use this when an NPC agrees to travel with the player.
        
        Args:
            npc_name: Name of the companion
            description: Brief description (role, personality, etc.)
            
        Returns:
            Confirmation message
        """
        companion = {
            "name": npc_name,
            "description": description,
            "joined_at_turn": self.turn_count
        }
        
        self.companions.append(companion)
        logger.info(f"👥 Companion joined: {npc_name}")
        
        return f"{npc_name} has joined as a companion"
    
    @function_tool
    async def record_clue(self, context: RunContext, clue: str, location: str):
        """Record a clue discovered during investigation (Detective universe).
        
        Use this when the player finds evidence, witnesses something, or discovers information.
        
        Args:
            clue: Description of the clue found
            location: Where the clue was found
            
        Returns:
            Confirmation message
        """
        clue_entry = {
            "clue": clue,
            "location": location,
            "turn": self.turn_count,
            "timestamp": datetime.now().isoformat()
        }
        
        self.clues.append(clue_entry)
        logger.info(f"🔍 Clue recorded: {clue}")
        
        return f"Clue recorded: {clue}"
    
    @function_tool
    async def add_suspect(
        self, 
        context: RunContext, 
        name: str, 
        description: str,
        motive: str = "Unknown",
        alibi: str = "Unknown"
    ):
        """Track suspects in the case (Detective universe).
        
        Use this when the player encounters a person of interest in the investigation.
        
        Args:
            name: Suspect's name
            description: Physical description and background
            motive: Their potential motive for the crime
            alibi: Their claimed whereabouts during the crime
            
        Returns:
            Confirmation message
        """
        suspect = {
            "name": name,
            "description": description,
            "motive": motive,
            "alibi": alibi,
            "added_at_turn": self.turn_count
        }
        
        self.suspects.append(suspect)
        logger.info(f"🕵️ Suspect added: {name}")
        
        return f"Added {name} to suspect list"
    
    @function_tool
    async def review_case_notes(self, context: RunContext):
        """Review all clues and suspects collected so far (Detective universe).
        
        Use this when player asks to review their notes, clues, or suspect list.
        
        Returns:
            Summary of all evidence and suspects
        """
        if not self.clues and not self.suspects:
            return "You haven't collected any clues or identified any suspects yet."
        
        notes = "CASE NOTES:\n\n"
        
        if self.clues:
            notes += "CLUES DISCOVERED:\n"
            for i, clue_entry in enumerate(self.clues, 1):
                notes += f"{i}. {clue_entry['clue']} (Found at: {clue_entry['location']})\n"
            notes += "\n"
        
        if self.suspects:
            notes += "SUSPECTS:\n"
            for i, suspect in enumerate(self.suspects, 1):
                notes += f"{i}. {suspect['name']}\n"
                notes += f"   Motive: {suspect['motive']}\n"
                notes += f"   Alibi: {suspect['alibi']}\n"
            notes += "\n"
        
        logger.info(f"📋 Case notes reviewed: {len(self.clues)} clues, {len(self.suspects)} suspects")
        
        return notes
    
    @function_tool
    async def save_session(self, context: RunContext, session_title: str):
        """Save the current game session to a JSON file.
        
        Use this when the player wants to end the session or at major story milestones.
        
        Args:
            session_title: A title for this session (e.g., "The Dragon's Lair")
            
        Returns:
            Confirmation with session details
        """
        session_id = f"SESSION_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        session_data = {
            "session_id": session_id,
            "title": session_title,
            "universe": self.universe,
            "tone": self.tone,
            "player_name": self.player_name,
            "total_turns": self.turn_count,
            "current_location": self.current_location,
            "inventory": self.inventory,
            "companions": [c['name'] for c in self.companions],
            "clues": self.clues if self.universe == "detective" else [],
            "suspects": [s['name'] for s in self.suspects] if self.universe == "detective" else [],
            "story_events": self.story_events,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to file
        sessions_dir = Path("game_sessions")
        sessions_dir.mkdir(exist_ok=True)
        
        session_file = sessions_dir / f"{session_id}.json"
        
        with open(session_file, "w") as f:
            json.dump(session_data, f, indent=2)
        
        logger.info(f"💾 Session saved: {session_id} - {session_title}")
        
        summary = f"Session '{session_title}' saved! You played for {self.turn_count} turns"
        if self.inventory:
            summary += f" and collected {len(self.inventory)} items"
        if self.companions:
            summary += f" with {len(self.companions)} companion(s)"
        
        return summary


def prewarm(proc: JobProcess):
    """Pre-load models to reduce latency"""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """Main entry point for the Game Master agent"""
    
    ctx.log_context_fields = {"room": ctx.room.name}
    
    # You can change these to customize the game!
    # Options: "fantasy", "sci-fi", "post-apocalypse", "horror", "detective"
    universe = "detective"
    # Options: "dramatic", "humorous", "spooky", "epic", "noir"
    tone = "dramatic"
    
    logger.info(f"🎲 Starting {tone} {universe} adventure")
    
    # Create voice pipeline
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",  # Deep, narrative voice
            style="Narration",      # Story-telling style
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )
    
    await session.start(
        agent=GameMasterAgent(universe=universe, tone=tone),
        room=ctx.room,
    )
    
    await ctx.connect()
    
    logger.info("🎲 Game Master ready to begin the adventure!")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))