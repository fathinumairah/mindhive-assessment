"""
Mock data for testing the API endpoints.
"""

# Mock outlet data
# https://zuscoffee.com/category/store/kuala-lumpur-selangor/
MOCK_OUTLETS = [
    {
        "name": "ZUS Coffee - SS2",
        "address": "5, Jalan SS 2/67, SS 2, 47300 Petaling Jaya, Selangor",
        "opening_time": "07:00",
        "closing_time": "21:40",
        "services": ["Dine-in", "Takeaway", "No-contact delivery"],
    },
    {
        "name": "ZUS Coffee - Uptown Damansara",
        "address": "44-G (Ground Floor, Jalan SS21/39, Damansara Utama, 47400 Petaling Jaya, Selangor",
        "opening_time": "07:00",
        "closing_time": "10:40",
        "services": ["Dine-in", "Takeaway", "No-contact delivery"],
    },
    {
        "name": "ZUS Coffee - Menara UOA Bangsar",
        "address": "Lot LGF-8, Lower Ground Floor, 5, Jalan Bangsar Utama 1, Bangsar, 59000 Kuala Lumpur, Wilayah Persekutuan Kuala Lumpur",
        "opening_time": "07:30",
        "closing_time": "19:40",
        "services": ["Dine-in", "Takeaway", "No-contact delivery"],
    },
    {
        "name": "ZUS Coffee - M3 Shopping Mall",
        "address": "GK-10a & GK-10b, Jln Madrasah, Taman Melati, 53100 Kuala Lumpur, Wilayah Persekutuan Kuala Lumpur",
        "opening_time": "10:00",
        "closing_time": "21:40",
        "services": ["Dine-in", "Takeaway"],
    }
]

# Mock product data
# https://shop.zuscoffee.com/
MOCK_PRODUCTS = [
    {
        "name": "ZUS All Day Cup 500ml (17oz) - Aqua Collection",
        "description": "When you're the life of the party, you'll find yourself gravitating towards experiences that can be shared with others. You're like the endless water: expansive, an adventure waiting to unfold. You're never in one place when there's a whole world out there. Bold, playful, and a thrill-seeker, you're best known by the Aqua colourways you carry. If you can relate to the collection, these cups are your perfect 'All Day' companion, keeping your drinks just the way you like them—hot or cold, up to 16 hours.",
        "price": 79.00,
        "colors": ["Misty Blue", "Ocean Breeze", "Blue Lagoon", "Deep Sea"],
        "category": "Drinkware"
    },
    {
        "name": "ZUS All Day Cup 500ml (17oz) - Mountain Collection",
        "description": "Maybe quietude and constancy are your mottos, or you feel at home amongst the fresh, clean air of higher altitudes. As steadfast as the tall trees, you've got a grounded confidence about you that speaks of surety, of someone who knows what they want and moves to achieve it. Strategic, mature, and observant, you're best defined by the Mountain colourways. If you can relate to the collection, these cups are your perfect 'All Day' companion, keeping your drinks just the way you like them—hot or cold, up to 16 hours.",
        "price": 79.00,
        "colors": ["Soft Fern", "Pine Green", "Terrain Green", "Forest Green"],
        "category": "Drinkware"
    },
    {
        "name": "ZUS All-Can Tumbler 600ml (20oz)",
        "description": "Dive into any sip-tuation with the all-new ZUS All-Can Tumbler, now more versatile than ever in Thunder Blue and Stainless Steel. Featuring interchangeable lids, use the “Screw On” lid for your hot Americano at work in the morning, and the “Flip Top” lid for easy hydration at the gym in the evening. Whether youre trapped in a jungle or an office, All-Can is with you through it all.",
        "price": 105.00,
        "colors": ["Thunder Blue", "Stainless Steel"],
        "category": "Drinkware"
    },
    {
        "name": "ZUS OG Ceramic Mug (16oz)",
        "description": "Designed for your cozy moments, our high-quality ceramic mug blends style and practicality effortlessly with an ergonomic handle that makes every sip a comfortable experience. Who said sipping from a mug couldn't be stylish? Now available in 3 of your favourite colours: Thunder Blue, Space Black, and Cloud White.",
        "price": 39.00,
        "colors": ["Thunder Blue", "Cloud White", "Space Black"],
        "category": "Drinkware"
    }
] 