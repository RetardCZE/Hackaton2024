base_texts = [
    "Cats are great pets and they love to climb trees",
    "Machine learning models can be very complex",
    "OpenAI creates advanced AI technologies",
    "The sun sets beautifully over the mountains",
    "JavaScript is widely used for web development",
    "Cooking healthy meals can improve your life",
    "The beach is a perfect place to relax",
    "Classical music can be very soothing",
    "A balanced diet is important for health",
    "Exploring new countries can be an adventure",
    "Birds are fascinating creatures with the ability to fly",
    "Mathematics is the language of the universe",
    "Robots are becoming more common in manufacturing",
    "The night sky is filled with stars and constellations",
    "HTML and CSS are the building blocks of the web",
    "Exercising regularly can boost your energy levels",
    "Mountains offer breathtaking views and fresh air",
    "Jazz music has a rich history and is very expressive",
    "A good night's sleep is essential for well-being",
    "Traveling can expand your perspective on life",
    "Dogs are known for their loyalty and companionship",
    "Physics explores the fundamental laws of nature",
    "AI can assist in various fields such as healthcare",
    "The ocean is vast and full of marine life",
    "React is a popular library for building user interfaces",
    "Gardening can be a relaxing and rewarding hobby",
    "Reading fiction can enhance your creativity",
    "The forest is home to diverse flora and fauna",
    "Blues music conveys deep emotions and stories",
    "Meditation can reduce stress and increase mindfulness",
    "Learning new languages can be a fulfilling challenge",
    "Hiking trails offer opportunities to connect with nature",
    "Chemistry helps us understand the composition of matter",
    "Blockchain technology underpins cryptocurrencies",
    "The desert has its own unique beauty and ecosystem",
    "Hip-hop culture includes music, dance, and art",
    "A strong immune system is key to staying healthy",
    "Experiencing different cultures can enrich your life",
    "Horseback riding can be a thrilling activity",
    "Biology studies the complexity of living organisms",
    "Data science combines statistics and programming",
    "Rainforests are crucial for the planet's biodiversity",
    "Electronic music spans a wide range of genres",
    "Yoga practice can improve flexibility and mental clarity",
    "Space exploration advances our understanding of the cosmos",
    "Chess is a strategic game that requires critical thinking",
    "The tundra is a unique and harsh biome",
    "Pop music often reflects contemporary trends",
    "A nutritious breakfast sets the tone for the day",
    "Urban exploration reveals hidden aspects of cities",
    "Ballet is a graceful and disciplined form of dance",
    "Astronomy allows us to observe distant galaxies",
    "Software development is a constantly evolving field",
    "Kayaking on a calm lake can be very peaceful",
    "Genetics helps us understand heredity and variation",
    "Augmented reality blends digital and physical worlds",
    "Caving is an adventurous way to explore underground",
    "Contemporary art challenges traditional boundaries",
    "Cycling is a great way to stay fit and explore",
    "The coral reef is a vibrant underwater ecosystem",
    "Classical literature offers timeless insights",
    "Birdwatching can be a relaxing and educational activity",
    "Astronauts train rigorously for space missions",
    "Self-driving cars are an emerging technology",
    "The polar regions are home to unique wildlife",
    "Country music often tells stories of life and love",
    "Healthy relationships are based on mutual respect",
    "Aquariums provide a window into marine environments",
    "Typography is a key aspect of graphic design",
    "Mountaineering requires skill and physical endurance",
    "The savanna is characterized by open grasslands",
    "Opera combines music, drama, and visual arts",
    "A balanced lifestyle includes work, rest, and play",
    "Theater productions bring stories to life on stage",
    "Zoology studies animal behavior and biology",
    "The internet connects people around the world",
    "Crafting can be a creative and satisfying pastime",
    "The wetland is an important ecological zone",
    "Piano playing can be both relaxing and challenging",
    "Cryptography is essential for securing digital communication",
    "The galaxy is a vast and mysterious place",
    "Photography captures moments in time",
    "A healthy work-life balance is crucial for well-being",
    "Painting allows for personal expression through art",
    "Renewable energy sources are vital for sustainability",
    "The steppe is a large area of flat unforested grassland",
    "Classical ballet requires years of training",
    "Virtual reality creates immersive digital experiences",
    "The rainforest canopy is teeming with life",
    "Playing the guitar can be a fun and rewarding hobby",
    "Cybersecurity protects information and systems",
    "Mountain biking offers both challenge and adventure",
    "Marine biology studies ocean ecosystems",
    "Artificial neural networks mimic the human brain",
    "The savanna is home to diverse wildlife",
    "Sculpture is a three-dimensional form of art",
    "Digital marketing is crucial for modern businesses",
    "The alpine biome is characterized by cold temperatures",
    "Studying history helps us understand the present",
    "Voice acting brings animated characters to life",
    "The swamp is a wetland area with rich biodiversity",
    "Playing the violin requires precision and practice",
    "Cloud computing offers scalable digital resources",
    "The prairie is known for its tall grasses and wildlife",
    "Architecture blends art and engineering",
    "Machine translation helps bridge language barriers",
    "The taiga is a forested biome with cold climates",
    "Singing can be a powerful form of self-expression",
    "The blockchain is a decentralized digital ledger",
    "Acupuncture is an ancient healing practice",
    "The grassland is a habitat with rich plant life",
    "Web design combines aesthetics and functionality",
    "Urban planning shapes the development of cities",
    "The Arctic is a polar region with extreme conditions",
    "Acting allows individuals to portray various characters",
    "Economics studies the production and consumption of goods",
    "The chaparral is a shrubland biome with hot, dry summers",
    "Creative writing lets individuals share their stories",
    "The marine biome covers the majority of the Earth's surface",
    "Interior design enhances the functionality of spaces",
    "Data visualization helps communicate complex information",
    "The temperate forest has four distinct seasons",
    "Sculpting with clay is a hands-on creative process",
    "Digital art uses technology as a medium",
    "Urban wildlife adapts to city environments",
    "Environmental science studies the impact of humans on nature",
    "The boreal forest is a dense, cold woodland",
    "Fashion design combines creativity with practicality",
    "The desert biome has extreme temperatures",
    "Learning to code can open up career opportunities",
    "The Mediterranean biome has hot, dry summers",
    "Ceramics involve creating objects from clay",
    "Game development is a multidisciplinary field",
    "The tropical rainforest has a high level of biodiversity",
    "Voice over work requires vocal skill and versatility",
    "The ocean biome is the largest and most diverse",
    "Storytelling is a fundamental human experience",
    "The temperate grassland is a fertile and productive biome",
    "Tattoo art is a form of personal expression",
    "The mangrove forest is a coastal ecosystem",
    "Podcasting allows individuals to share their ideas",
    "The savanna biome has seasonal rainfall patterns",
    "Illustration combines art with communication",
    "Video production involves multiple creative processes",
    "The river biome is a dynamic freshwater ecosystem",
    "Leatherworking is a traditional craft",
    "Augmented reality can enhance real-world experiences",
    "The island biome is isolated with unique species",
    "Songwriting allows for creative musical expression",
    "The coral reef is one of the most diverse ecosystems",
    "The rainforest floor is dark and damp",
    "Playing the drums requires rhythm and coordination",
    "The tundra biome is cold and treeless",
    "Knitting can be a relaxing and productive hobby",
    "The woodland biome has a variety of tree species",
    "Graphic design combines images and text",
    "The lagoon is a shallow body of water separated from the sea",
    "Baking involves both science and creativity",
    "The temperate rainforest has high precipitation",
    "Dance allows for physical and artistic expression",
    "The shrubland biome is adapted to dry conditions",
    "Jewelry making involves skill and creativity",
    "The deciduous forest has trees that lose their leaves",
    "Screenwriting involves crafting dialogue and plots",
    "The river delta is a fertile area where rivers meet the sea",
    "Pottery is an ancient form of art and utility",
    "The grassland biome is home to many grazing animals",
    "Performing arts include music, theater, and dance",
    "The bog is a wetland with acidic, nutrient-poor water",
    "The marsh is a wetland with grasses and reeds",
    "The coral atoll is a ring-shaped coral reef",
    "Floral design involves arranging flowers aesthetically",
    "The fen is a wetland with alkaline water",
    "The moor is a habitat with peat and heather",
    "Weaving involves creating fabric from threads",
    "The estuary is where freshwater meets saltwater",
    "Glassblowing is a technique for shaping glass",
    "The fjord is a long, narrow inlet with steep sides",
    "Basket weaving is a traditional craft",
    "The geyser is a hot spring that erupts periodically",
    "Mosaic art involves creating images with small pieces",
    "The ice cap biome is covered in ice and snow",
    "Metalworking involves shaping metal into objects",
    "The salt marsh is a coastal wetland",
    "Woodcarving is a form of sculpting wood",
    "The polar ice biome is characterized by ice and snow",
    "Quilting involves sewing layers of fabric together"
]
