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

logger = logging.getLogger("food-ordering-agent")
load_dotenv(".env.local")


class FoodOrderingAgent(Agent):
    def __init__(self, catalog_data: dict) -> None:
        self.catalog = catalog_data
        self.cart = []  # List of {item_id, name, quantity, price, total}
        self.customer_name = None
        
        super().__init__(
            instructions="""You are a friendly food and grocery ordering assistant for FreshMart Store.

YOUR ROLE:
- Help customers order groceries, snacks, and prepared foods
- Be helpful, patient, and conversational
- Confirm additions and changes to the cart clearly

GREETING:
"Hi! Welcome to FreshMart. I can help you order groceries, snacks, and meal ingredients. What would you like to order today?"

HOW TO HANDLE ORDERS:

1. SPECIFIC ITEMS:
   When customer says "I want bread" or "Add milk":
   - Use search_catalog tool to find the item
   - If multiple options (brands/sizes), ask which one they prefer
   - Ask for quantity if not specified
   - Use add_to_cart tool with item_id and quantity
   - Confirm: "I've added [quantity] [item name] to your cart."

2. RECIPE/MEAL REQUESTS:
   When customer says "I need ingredients for pasta" or "Get me what I need for a sandwich":
   - Use add_recipe_to_cart tool with the dish name
   - This will automatically add all needed ingredients
   - Confirm what was added: "I've added pasta, pasta sauce, and cheese for your pasta meal."

3. CART MANAGEMENT:
   - "What's in my cart?" → IMMEDIATELY use show_cart tool
   - "Remove [item]" → IMMEDIATELY use remove_from_cart tool with the item name
   - "Change [item] to [quantity]" → IMMEDIATELY use update_quantity tool with item name and new quantity
   - "Make it [number]" → IMMEDIATELY use update_quantity tool
   
   CRITICAL: Don't just acknowledge these requests - USE THE TOOLS IMMEDIATELY

4. CHECKOUT:
   When customer says "That's all", "Place my order", "I'm done", "Checkout":
   - Use show_cart to review everything
   - Ask for their name if you don't have it
   - Use place_order tool to save the order
   - Confirm: "Your order has been placed! Order number [ID]. Total: ₹[amount]. Thank you!"

IMPORTANT RULES:
- Always confirm what you're adding to the cart
- Ask clarifying questions when needed (brand, size, quantity)
- Be proactive: "Anything else you'd like to add?"
- Use tools immediately - don't just talk about using them
- Keep conversation natural and friendly
- No bullet points or formatting in speech""",
        )
    
    @function_tool
    async def search_catalog(self, context: RunContext, item_name: str):
        """Search for items in the catalog by name or tags.
        
        Use this when customer mentions an item they want to buy.
        
        Args:
            item_name: The item name or description (e.g., "bread", "milk", "chips")
            
        Returns:
            List of matching items with details
        """
        item_name_lower = item_name.lower()
        matches = []
        
        # Search in all categories
        for category in ['groceries', 'snacks', 'prepared_food']:
            if category in self.catalog:
                for item in self.catalog[category]:
                    # Check if search term is in name or tags
                    if (item_name_lower in item['name'].lower() or 
                        any(item_name_lower in tag.lower() for tag in item.get('tags', []))):
                        matches.append({
                            'id': item['id'],
                            'name': item['name'],
                            'brand': item.get('brand', 'N/A'),
                            'size': item.get('size', 'N/A'),
                            'price': item['price'],
                            'category': item['category']
                        })
        
        if matches:
            logger.info(f"Found {len(matches)} matches for '{item_name}'")
            if len(matches) == 1:
                return f"Found: {matches[0]['name']} ({matches[0]['brand']}, {matches[0]['size']}) - ₹{matches[0]['price']}. Item ID: {matches[0]['id']}"
            else:
                result = f"Found {len(matches)} options:\n"
                for idx, m in enumerate(matches, 1):
                    result += f"{idx}. {m['name']} ({m['brand']}, {m['size']}) - ₹{m['price']} [ID: {m['id']}]\n"
                return result
        else:
            logger.warning(f"No matches found for '{item_name}'")
            return f"Sorry, I couldn't find '{item_name}' in our catalog. Could you try describing it differently?"
    
    @function_tool
    async def add_to_cart(self, context: RunContext, item_id: str, quantity: int = 1):
        """Add a specific item to the cart.
        
        Use this after searching and finding the item the customer wants.
        
        Args:
            item_id: The item ID (e.g., "G001", "S002")
            quantity: How many of this item (default: 1)
            
        Returns:
            Confirmation message
        """
        # Find item in catalog
        item = self._find_item_by_id(item_id)
        
        if not item:
            return f"Error: Item {item_id} not found in catalog"
        
        # Check if item already in cart
        for cart_item in self.cart:
            if cart_item['item_id'] == item_id:
                cart_item['quantity'] += quantity
                cart_item['total'] = cart_item['quantity'] * cart_item['price']
                logger.info(f"Updated {item['name']} quantity to {cart_item['quantity']}")
                return f"Updated cart: {cart_item['quantity']} x {item['name']} (₹{cart_item['total']})"
        
        # Add new item to cart
        cart_item = {
            'item_id': item_id,
            'name': item['name'],
            'brand': item.get('brand', 'N/A'),
            'size': item.get('size', 'N/A'),
            'quantity': quantity,
            'price': item['price'],
            'total': item['price'] * quantity
        }
        self.cart.append(cart_item)
        
        logger.info(f"Added to cart: {quantity} x {item['name']}")
        return f"Added {quantity} x {item['name']} (₹{cart_item['total']}) to your cart"
    
    @function_tool
    async def add_recipe_to_cart(self, context: RunContext, dish_name: str):
        """Add all ingredients for a specific dish/recipe to the cart.
        
        Use this when customer asks for "ingredients for X" or "what I need for X".
        Examples: "pasta", "sandwich", "breakfast"
        
        Args:
            dish_name: The dish/meal name (e.g., "pasta", "peanut butter sandwich")
            
        Returns:
            Confirmation of what was added
        """
        dish_name_lower = dish_name.lower()
        
        # Find matching recipe
        if 'recipes' not in self.catalog:
            return "Sorry, I don't have recipe information available"
        
        recipe_found = None
        for recipe_key, item_ids in self.catalog['recipes'].items():
            if dish_name_lower in recipe_key.lower() or recipe_key in dish_name_lower:
                recipe_found = item_ids
                break
        
        if not recipe_found:
            return f"Sorry, I don't have a recipe for '{dish_name}'. Try asking for specific items instead."
        
        # Add all items from recipe
        added_items = []
        for item_id in recipe_found:
            item = self._find_item_by_id(item_id)
            if item:
                # Add to cart
                await self.add_to_cart(context, item_id, quantity=1)
                added_items.append(item['name'])
        
        logger.info(f"Added recipe '{dish_name}': {added_items}")
        return f"Added ingredients for {dish_name}: {', '.join(added_items)}"
    
    @function_tool
    async def show_cart(self, context: RunContext):
        """Show all items currently in the cart.
        
        Use this when customer asks "What's in my cart?" or before checkout.
        
        Returns:
            Cart summary with items and total
        """
        if not self.cart:
            return "Your cart is empty. What would you like to order?"
        
        cart_summary = "Here's what's in your cart:\n"
        total = 0
        
        for item in self.cart:
            cart_summary += f"- {item['quantity']} x {item['name']} ({item['brand']}, {item['size']}) = ₹{item['total']}\n"
            total += item['total']
        
        cart_summary += f"\nTotal: ₹{total:.2f}"
        
        logger.info(f"Cart summary: {len(self.cart)} items, ₹{total}")
        return cart_summary
    
    @function_tool
    async def remove_from_cart(self, context: RunContext, item_name: str):
        """Remove an item from the cart.
        
        Use this when customer says "Remove [item]" or "Take out [item]".
        
        Args:
            item_name: Name of the item to remove (can be partial name)
            
        Returns:
            Confirmation message
        """
        if not self.cart:
            return "Your cart is empty."
        
        item_name_lower = item_name.lower().strip()
        
        for i, cart_item in enumerate(self.cart):
            item_full_name = cart_item['name'].lower()
            # Check if search term is in the item name or vice versa
            if item_name_lower in item_full_name or item_full_name in item_name_lower:
                removed = self.cart.pop(i)
                logger.info(f"Removed from cart: {removed['name']}")
                return f"Removed {removed['name']} from your cart"
        
        # If no match, show what's in cart
        cart_items = [item['name'] for item in self.cart]
        return f"I couldn't find '{item_name}' in your cart. You have: {', '.join(cart_items)}"
    
    @function_tool
    async def update_quantity(self, context: RunContext, item_name: str, new_quantity: int):
        """Update the quantity of an item in the cart.
        
        Use this when customer says "Change [item] to [number]" or "Make it [number]".
        
        Args:
            item_name: Name of the item to update (can be partial name)
            new_quantity: New quantity (if 0, remove item)
            
        Returns:
            Confirmation message
        """
        logger.info(f"🔍 UPDATE REQUEST: item_name='{item_name}', new_quantity={new_quantity}")
        logger.info(f"🛒 Current cart: {[item['name'] for item in self.cart]}")
        
        if new_quantity == 0:
            return await self.remove_from_cart(context, item_name)
        
        if not self.cart:
            return "Your cart is empty."
        
        item_name_lower = item_name.lower().strip()
        
        # Try to find matching item
        matched_item = None
        for cart_item in self.cart:
            item_full_name = cart_item['name'].lower()
            # Check if search term is in the item name or vice versa
            if item_name_lower in item_full_name or item_full_name in item_name_lower:
                matched_item = cart_item
                break
        
        if matched_item:
            old_qty = matched_item['quantity']
            matched_item['quantity'] = new_quantity
            matched_item['total'] = matched_item['price'] * new_quantity
            logger.info(f"✅ Updated {matched_item['name']}: {old_qty} → {new_quantity}")
            return f"Updated {matched_item['name']} quantity from {old_qty} to {new_quantity}. New total: ₹{matched_item['total']}"
        
        # If no match, show what's in cart
        cart_items = [item['name'] for item in self.cart]
        logger.warning(f"❌ No match found for '{item_name}'")
        return f"I couldn't find '{item_name}' in your cart. You have: {', '.join(cart_items)}"
    
    @function_tool
    async def place_order(self, context: RunContext, customer_name: str):
        """Place the final order and save it to a JSON file.
        
        Use this when customer is done and says "Place order", "That's all", "Checkout".
        
        Args:
            customer_name: Customer's name for the order
            
        Returns:
            Order confirmation with order number
        """
        if not self.cart:
            return "Your cart is empty. Add some items before placing an order."
        
        self.customer_name = customer_name
        
        # Calculate total
        total = sum(item['total'] for item in self.cart)
        
        # Create order object
        order_id = f"ORD_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        order = {
            'order_id': order_id,
            'customer_name': customer_name,
            'timestamp': datetime.now().isoformat(),
            'items': self.cart,
            'total': total,
            'currency': 'INR',
            'status': 'placed'
        }
        
        # Save to JSON file
        orders_dir = Path("orders")
        orders_dir.mkdir(exist_ok=True)
        
        order_file = orders_dir / f"{order_id}.json"
        
        with open(order_file, "w") as f:
            json.dump(order, f, indent=2)
        
        logger.info(f"✓ Order placed: {order_id} for {customer_name}, Total: ₹{total}")
        
        return f"Order confirmed! Your order number is {order_id}. Total: ₹{total:.2f}. Thank you for shopping with FreshMart!"
    
    def _find_item_by_id(self, item_id: str):
        """Helper method to find item in catalog by ID"""
        for category in ['groceries', 'snacks', 'prepared_food']:
            if category in self.catalog:
                for item in self.catalog[category]:
                    if item['id'] == item_id:
                        return item
        return None


def load_catalog():
    """Load catalog from JSON file"""
    catalog_file = Path("DB/catalog.json")
    
    if not catalog_file.exists():
        logger.error("catalog.json not found!")
        return None
    
    with open(catalog_file, "r") as f:
        return json.load(f)


def prewarm(proc: JobProcess):
    """Pre-load models to reduce latency"""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """Main entry point for the food ordering agent"""
    
    ctx.log_context_fields = {"room": ctx.room.name}
    
    # Load catalog
    catalog = load_catalog()
    if not catalog:
        logger.error("Failed to load catalog. Exiting.")
        return
    
    logger.info(f"Catalog loaded: {len(catalog.get('groceries', []))} groceries, "
                f"{len(catalog.get('snacks', []))} snacks, "
                f"{len(catalog.get('prepared_food', []))} prepared foods")
    
    # Create voice pipeline
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",  # Indian female voice
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )
    
    await session.start(
        agent=FoodOrderingAgent(catalog),
        room=ctx.room,
    )
    
    await ctx.connect()
    
    logger.info("🛒 Food Ordering Agent ready!")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))