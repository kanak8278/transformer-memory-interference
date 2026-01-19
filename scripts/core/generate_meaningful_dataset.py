"""
Generate interleaved dataset with EXPANDED meaningful real-world values.

ALL 46 categories expanded to 50-70+ values each for testing up to interference level 50-70.
"""

import json
import random
from typing import Dict, List
from pathlib import Path


# Define EXPANDED meaningful value pools for each category (50-70+ values each)
CATEGORY_VALUES = {
    "visual art": [
        # Major art movements (20)
        "impressionism", "cubism", "surrealism", "abstract expressionism", "pop art",
        "baroque", "renaissance", "romanticism", "realism", "modernism",
        "minimalism", "dadaism", "fauvism", "pointillism", "art deco",
        "constructivism", "futurism", "expressionism", "neoclassicism", "rococo",
        # Additional movements (20)
        "mannerism", "symbolism", "post impressionism", "art nouveau", "bauhaus",
        "suprematism", "vorticism", "orphism", "rayonism", "precisionism",
        "neue sachlichkeit", "social realism", "magic realism", "neo expressionism", "transavantgarde",
        "photorealism", "op art", "kinetic art", "conceptual art", "performance art",
        # Contemporary & Regional (20)
        "street art", "lowbrow", "stuckism", "young british artists", "neo pop",
        "superflat", "chinese contemporary", "african modernism", "indigenous art", "outsider art",
        "graffiti art", "digital art", "new media art", "bio art", "land art",
        "installation art", "video art", "sound art", "light art", "environmental art"
    ],

    "tools": [
        # Hand tools - Striking (15)
        "hammer", "mallet", "sledgehammer", "ball peen hammer", "claw hammer",
        "rubber mallet", "dead blow hammer", "club hammer", "framing hammer", "tack hammer",
        "brick hammer", "welding hammer", "engineer hammer", "cross peen hammer", "chipping hammer",
        # Hand tools - Cutting (15)
        "saw", "hacksaw", "handsaw", "coping saw", "bow saw",
        "pruning saw", "keyhole saw", "backsaw", "japanese saw", "compass saw",
        "wire cutters", "bolt cutters", "tin snips", "aviation snips", "pipe cutter",
        # Hand tools - Gripping/Turning (15)
        "wrench", "pliers", "screwdriver", "allen key", "socket wrench",
        "adjustable wrench", "pipe wrench", "torque wrench", "needle nose pliers", "locking pliers",
        "flathead screwdriver", "phillips screwdriver", "ratchet", "box wrench", "combination wrench",
        # Power tools & Precision (15)
        "drill", "impact driver", "circular saw", "jigsaw", "angle grinder",
        "sander", "router", "nail gun", "staple gun", "heat gun",
        "chisel", "file", "rasp", "plane", "spokeshave",
        # Measuring & Other (10)
        "tape measure", "level", "square", "caliper", "micrometer",
        "clamp", "vise", "crowbar", "pry bar", "punch"
    ],

    "landform": [
        # Basic landforms (15)
        "mountain", "valley", "plateau", "canyon", "plain",
        "hill", "dune", "cliff", "peninsula", "island",
        "glacier", "volcano", "mesa", "butte", "ridge",
        # Water-related landforms (10)
        "delta", "fjord", "lagoon", "estuary", "strait",
        "bay", "cove", "inlet", "sound", "gulf",
        # Elevated landforms (10)
        "peak", "summit", "foothill", "escarpment", "bluff",
        "tor", "crag", "precipice", "knoll", "mound",
        # Erosional features (10)
        "gorge", "ravine", "gully", "arroyo", "wash",
        "karst", "sinkhole", "cave", "grotto", "tunnel",
        # Depositional features (10)
        "moraine", "drumlin", "esker", "kame", "outwash",
        "alluvial fan", "floodplain", "terrace", "levee", "bar"
    ],

    "musical instrument": [
        # String instruments (20)
        "guitar", "piano", "violin", "cello", "harp",
        "banjo", "mandolin", "ukulele", "sitar", "lute",
        "zither", "dulcimer", "balalaika", "koto", "guqin",
        "viola", "double bass", "harpsichord", "lyre", "oud",
        # Wind instruments - Woodwinds (15)
        "flute", "clarinet", "oboe", "saxophone", "bassoon",
        "piccolo", "recorder", "panpipes", "harmonica", "melodica",
        "english horn", "contrabassoon", "bass clarinet", "shakuhachi", "didgeridoo",
        # Wind instruments - Brass (10)
        "trumpet", "trombone", "french horn", "tuba", "cornet",
        "euphonium", "bugle", "flugelhorn", "sousaphone", "mellophone",
        # Percussion (15)
        "drums", "timpani", "xylophone", "marimba", "vibraphone",
        "glockenspiel", "tambourine", "cymbals", "gong", "chimes",
        "castanets", "triangle", "cowbell", "bongos", "congas",
        # Other/Electronic (10)
        "accordion", "bagpipes", "organ", "synthesizer", "theremin",
        "steel drum", "djembe", "tabla", "kalimba", "ocarina"
    ],

    "gemstone": [
        # Precious stones (10)
        "diamond", "ruby", "sapphire", "emerald", "alexandrite",
        "padparadscha", "red beryl", "tanzanite", "black opal", "jadeite",
        # Semi-precious - Quartz family (15)
        "amethyst", "citrine", "rose quartz", "smoky quartz", "tiger eye",
        "aventurine", "carnelian", "agate", "chalcedony", "jasper",
        "bloodstone", "chrysoprase", "onyx", "sardonyx", "prasiolite",
        # Semi-precious - Beryl family (8)
        "aquamarine", "morganite", "heliodor", "goshenite", "bixbite",
        "golden beryl", "green beryl", "pectolite",
        # Semi-precious - Other minerals (17)
        "topaz", "garnet", "peridot", "tourmaline", "spinel",
        "zircon", "kunzite", "moonstone", "sunstone", "labradorite",
        "lapis lazuli", "malachite", "azurite", "rhodochrosite", "rhodonite",
        "turquoise", "chrysocolla",
        # Organic gems (10)
        "pearl", "amber", "coral", "jet", "ivory",
        "mother of pearl", "ammolite", "shell", "bone", "copal"
    ],

    "fabric": [
        # Natural fibers (15)
        "cotton", "silk", "wool", "linen", "hemp",
        "cashmere", "mohair", "alpaca", "angora", "ramie",
        "jute", "sisal", "bamboo", "flax", "kapok",
        # Synthetic fibers (12)
        "polyester", "nylon", "acrylic", "spandex", "rayon",
        "acetate", "viscose", "lyocell", "modal", "elastane",
        "microfiber", "kevlar",
        # Blends & Weaves (18)
        "denim", "velvet", "satin", "chiffon", "tweed",
        "corduroy", "flannel", "jersey", "muslin", "taffeta",
        "organza", "brocade", "damask", "chenille", "tulle",
        "canvas", "burlap", "felt",
        # Technical fabrics (10)
        "gore-tex", "fleece", "terry cloth", "mesh", "ripstop",
        "oxford", "chambray", "poplin", "voile", "gauze"
    ],

    "tree species": [
        # Deciduous hardwoods (20)
        "oak", "maple", "birch", "ash", "elm",
        "beech", "poplar", "walnut", "cherry", "hickory",
        "mahogany", "teak", "ebony", "rosewood", "sycamore",
        "alder", "aspen", "basswood", "buckeye", "chestnut",
        # Coniferous softwoods (15)
        "pine", "cedar", "spruce", "fir", "redwood",
        "hemlock", "cypress", "juniper", "larch", "sequoia",
        "yew", "douglas fir", "balsam fir", "white pine", "ponderosa pine",
        # Tropical & Exotic (15)
        "palm", "bamboo", "eucalyptus", "acacia", "baobab",
        "banyan", "breadfruit", "cacao", "cashew", "cinnamon",
        "fig", "jacaranda", "mango", "neem", "sandalwood"
    ],

    "cheese variety": [
        # Hard cheeses (15)
        "cheddar", "parmesan", "gouda", "gruyere", "manchego",
        "asiago", "pecorino", "aged provolone", "emmental", "jarlsberg",
        "comte", "mimolette", "romano", "aged swiss", "cantal",
        # Soft cheeses (15)
        "brie", "camembert", "feta", "ricotta", "cream cheese",
        "boursin", "chevre", "mascarpone", "mozzarella", "burrata",
        "cottage cheese", "neufchatel", "quark", "paneer", "halloumi",
        # Blue cheeses (10)
        "roquefort", "gorgonzola", "stilton", "danish blue", "cambozola",
        "maytag blue", "cashel blue", "fourme d'ambert", "cabrales", "valdeon",
        # Semi-soft & Washed rind (15)
        "havarti", "monterey jack", "muenster", "fontina", "port salut",
        "taleggio", "raclette", "limburger", "epoisses", "pont l'eveque",
        "reblochon", "morbier", "saint nectaire", "tomme", "colby"
    ],

    "architectural style": [
        # Historical (20)
        "gothic", "baroque", "neoclassical", "romanesque", "byzantine",
        "rococo", "renaissance", "colonial", "victorian", "georgian",
        "tudor", "elizabethan", "jacobean", "palladian", "federal",
        "greek revival", "romanesque revival", "gothic revival", "beaux arts", "second empire",
        # Modern (20)
        "modernist", "brutalist", "bauhaus", "minimalist", "postmodern",
        "deconstructivism", "high tech", "international style", "streamline moderne", "art deco",
        "prairie style", "craftsman", "bungalow", "ranch", "split level",
        "mid century modern", "metabolism", "futurism", "expressionism", "constructivism",
        # Contemporary (10)
        "parametric", "sustainable", "green", "biomimetic", "digital",
        "neo futurism", "blob architecture", "critical regionalism", "new urbanism", "vernacular"
    ],

    "cloud formation": [
        # Basic types (10)
        "cumulus", "stratus", "cirrus", "nimbus", "cumulonimbus",
        "altostratus", "cirrocumulus", "stratocumulus", "altocumulus", "cirrostratus",
        # Specialized (15)
        "nimbostratus", "contrail", "mammatus", "lenticular", "pileus",
        "arcus", "kelvin helmholtz", "undulatus", "asperitas", "virga",
        "pyrocumulus", "flammagenitus", "fractus", "humilis", "mediocris",
        # Rare & Exotic (20)
        "congestus", "castellanus", "fibratus", "uncinus", "spissatus",
        "nebulosus", "translucidus", "perlucidus", "opacus", "duplicatus",
        "radiatus", "lacunosus", "cavum", "murus", "cauda",
        "velum", "pannus", "tuba", "incus", "praecipitatio"
    ],

    "bird species": [
        # Songbirds (15)
        "sparrow", "robin", "cardinal", "bluejay", "mockingbird",
        "warbler", "finch", "thrush", "oriole", "grosbeak",
        "bunting", "tanager", "wren", "chickadee", "nuthatch",
        # Birds of prey (12)
        "eagle", "hawk", "falcon", "owl", "vulture",
        "kite", "osprey", "condor", "buzzard", "harrier",
        "kestrel", "merlin",
        # Water birds (13)
        "duck", "goose", "swan", "heron", "egret",
        "pelican", "cormorant", "albatross", "gull", "tern",
        "loon", "grebe", "coot",
        # Other (15)
        "crow", "raven", "magpie", "woodpecker", "hummingbird",
        "flamingo", "peacock", "penguin", "ostrich", "emu",
        "kingfisher", "toucan", "parrot", "pigeon", "dove"
    ],

    "culinary herb": [
        # Mediterranean (15)
        "basil", "oregano", "thyme", "rosemary", "parsley",
        "sage", "marjoram", "bay leaf", "fennel", "lavender",
        "savory", "hyssop", "lovage", "sorrel", "borage",
        # Asian (12)
        "cilantro", "lemongrass", "thai basil", "shiso", "curry leaf",
        "kaffir lime", "vietnamese mint", "perilla", "gotu kola", "holy basil",
        "saw leaf", "pandan",
        # Onion family (8)
        "chives", "garlic chives", "scallion", "leek", "shallot",
        "ramps", "welsh onion", "spring onion",
        # Other (15)
        "dill", "mint", "tarragon", "chervil", "anise",
        "caraway", "coriander", "cumin", "fenugreek", "mustard",
        "watercress", "arugula", "celery leaf", "parsnip leaf", "horseradish"
    ],

    "flower species": [
        # Popular garden flowers (20)
        "rose", "tulip", "daisy", "sunflower", "lily",
        "orchid", "carnation", "daffodil", "iris", "peony",
        "chrysanthemum", "lavender", "hibiscus", "jasmine", "magnolia",
        "geranium", "marigold", "zinnia", "petunia", "snapdragon",
        # Spring bulbs (10)
        "crocus", "hyacinth", "narcissus", "snowdrop", "bluebell",
        "anemone", "ranunculus", "freesia", "gladiolus", "allium",
        # Perennials (15)
        "hydrangea", "azalea", "rhododendron", "camellia", "begonia",
        "fuchsia", "impatiens", "salvia", "verbena", "aster",
        "coneflower", "black eyed susan", "coreopsis", "delphinium", "foxglove",
        # Wildflowers (10)
        "poppy", "lupine", "buttercup", "clover", "dandelion",
        "columbine", "forget me not", "primrose", "violet", "wildflower"
    ],

    "wine variety": [
        # Red grapes (18)
        "cabernet sauvignon", "merlot", "pinot noir", "syrah", "malbec",
        "zinfandel", "sangiovese", "tempranillo", "grenache", "nebbiolo",
        "barbera", "gamay", "mourvedre", "petite sirah", "carmenere",
        "pinotage", "nero d'avola", "primitivo",
        # White grapes (18)
        "chardonnay", "sauvignon blanc", "riesling", "pinot grigio", "viognier",
        "gewurztraminer", "chenin blanc", "semillon", "marsanne", "roussanne",
        "albarino", "gruner veltliner", "torrontes", "vermentino", "verdejo",
        "fiano", "greco", "cortese",
        # Other (14)
        "moscato", "prosecco", "champagne", "cava", "port",
        "sherry", "madeira", "tokaji", "ice wine", "sauternes",
        "muscat", "lambrusco", "vin santo", "malmsey"
    ],

    "dance style": [
        # Ballroom (12)
        "waltz", "foxtrot", "tango", "quickstep", "viennese waltz",
        "cha cha", "rumba", "samba", "jive", "paso doble",
        "bolero", "mambo",
        # Latin (10)
        "salsa", "merengue", "bachata", "cumbia", "reggaeton",
        "zouk", "kizomba", "lambada", "forró", "tango argentino",
        # Modern (15)
        "hip hop", "breakdancing", "popping", "locking", "krumping",
        "house", "waacking", "voguing", "electro", "freestyle",
        "contemporary", "lyrical", "modern", "postmodern", "contact improvisation",
        # Traditional (13)
        "ballet", "tap", "jazz", "flamenco", "irish step",
        "clog", "folk", "square dance", "line dance", "swing",
        "charleston", "lindy hop", "jitterbug"
    ],

    "pasta shape": [
        # Long pasta (15)
        "spaghetti", "linguine", "fettuccine", "tagliatelle", "pappardelle",
        "angel hair", "bucatini", "vermicelli", "capellini", "bigoli",
        "bavette", "stringozzi", "trenette", "pizzoccheri", "mafalde",
        # Short pasta (20)
        "penne", "rigatoni", "fusilli", "farfalle", "rotini",
        "ziti", "macaroni", "orecchiette", "cavatappi", "gemelli",
        "campanelle", "radiatore", "strozzapreti", "casarecce", "garganelli",
        "ditalini", "orzo", "pastina", "acini di pepe", "stelline",
        # Filled pasta (10)
        "ravioli", "tortellini", "cappelletti", "agnolotti", "mezzelune",
        "cannelloni", "manicotti", "conchiglioni", "lumaconi", "tortelloni",
        # Other (10)
        "lasagna", "gnocchi", "spaetzle", "couscous", "fregula",
        "trofie", "busiate", "corzetti", "malloreddus", "lorighittas"
    ],

    "literary genre": [
        # Fiction genres (20)
        "science fiction", "fantasy", "mystery", "thriller", "horror",
        "romance", "historical fiction", "western", "adventure", "dystopian",
        "utopian", "magical realism", "urban fantasy", "steampunk", "cyberpunk",
        "space opera", "hard sci fi", "soft sci fi", "sword and sorcery", "dark fantasy",
        # Mystery subgenres (10)
        "detective", "crime", "noir", "cozy mystery", "police procedural",
        "legal thriller", "spy thriller", "techno thriller", "psychological thriller", "whodunit",
        # Literary styles (15)
        "satire", "gothic", "absurdist", "existentialist", "modernist",
        "postmodernist", "metafiction", "experimental", "magical realism", "surrealism",
        "allegory", "fable", "parable", "bildungsroman", "epistolary",
        # Non-fiction (10)
        "biography", "autobiography", "memoir", "essay", "poetry",
        "drama", "travelogue", "journalism", "philosophy", "history"
    ],

    "cooking method": [
        # Dry heat (15)
        "baking", "roasting", "grilling", "broiling", "searing",
        "pan frying", "deep frying", "stir frying", "sauteing", "smoking",
        "toasting", "caramelizing", "charring", "flambeing", "tempering",
        # Moist heat (12)
        "boiling", "simmering", "poaching", "steaming", "braising",
        "stewing", "blanching", "parboiling", "pressure cooking", "sous vide",
        "en papillote", "confit",
        # Combination (8)
        "browning", "deglazing", "reduction", "emulsifying", "marinating",
        "brining", "curing", "fermenting",
        # Baking techniques (10)
        "creaming", "folding", "kneading", "proofing", "laminating",
        "blind baking", "glazing", "piping", "tempering chocolate", "clarifying"
    ],

    "mathematical concept": [
        # Branches (15)
        "algebra", "geometry", "calculus", "trigonometry", "statistics",
        "probability", "topology", "number theory", "combinatorics", "graph theory",
        "set theory", "linear algebra", "differential equations", "complex analysis", "real analysis",
        # Operations (12)
        "addition", "subtraction", "multiplication", "division", "exponentiation",
        "logarithm", "derivative", "integral", "limit", "series",
        "sequence", "function",
        # Structures (13)
        "matrix", "vector", "tensor", "group", "ring",
        "field", "module", "lattice", "manifold", "space",
        "dimension", "symmetry", "transformation",
        # Numbers (10)
        "prime number", "composite number", "rational number", "irrational number", "real number",
        "complex number", "imaginary number", "transcendental number", "fraction", "polynomial"
    ],

    "weather phenomenon": [
        # Precipitation (12)
        "rain", "snow", "sleet", "hail", "freezing rain",
        "drizzle", "graupel", "ice pellets", "virga", "diamond dust",
        "snow squall", "rain shower",
        # Storms (15)
        "thunderstorm", "tornado", "hurricane", "typhoon", "cyclone",
        "blizzard", "ice storm", "derecho", "squall", "microburst",
        "supercell", "waterspout", "dust devil", "haboob", "nor'easter",
        # Optical (10)
        "rainbow", "aurora", "halo", "sun dog", "moon dog",
        "corona", "glory", "fog bow", "light pillar", "green flash",
        # Other (13)
        "fog", "mist", "haze", "smog", "frost",
        "dew", "drought", "heat wave", "cold snap", "wind chill",
        "lightning", "thunder", "monsoon"
    ],

    "ocean current": [
        # Major currents (15)
        "gulf stream", "kuroshio", "labrador", "california", "canary",
        "benguela", "agulhas", "humboldt", "antarctic circumpolar", "north atlantic",
        "north pacific", "south atlantic", "south pacific", "indian ocean", "arctic ocean",
        # Regional currents (20)
        "equatorial", "alaska", "east australian", "brazil", "peru",
        "west wind drift", "north equatorial", "south equatorial", "equatorial counter",
        "cromwell", "leeuwin", "oyashio", "somali", "mozambique", "east greenland",
        "west greenland", "norwegian", "irminger", "azores", "guinea",
        # Upwelling & Other (15)
        "upwelling", "downwelling", "thermohaline circulation", "meridional overturning",
        "gyre", "eddy", "ring", "filament", "jet", "meander",
        "convergence", "divergence", "coastal current", "boundary current", "western intensification"
    ],

    "mineral type": [
        # Silicates (15)
        "quartz", "feldspar", "mica", "olivine", "pyroxene",
        "amphibole", "garnet", "topaz", "tourmaline", "beryl",
        "zircon", "kyanite", "sillimanite", "andalusite", "epidote",
        # Carbonates (8)
        "calcite", "dolomite", "aragonite", "magnesite", "siderite",
        "rhodochrosite", "smithsonite", "cerussite",
        # Oxides (10)
        "hematite", "magnetite", "corundum", "ilmenite", "chromite",
        "rutile", "cassiterite", "pyrolusite", "cuprite", "spinel",
        # Sulfides (10)
        "pyrite", "galena", "sphalerite", "chalcopyrite", "cinnabar",
        "arsenopyrite", "molybdenite", "bornite", "pentlandite", "pyrrhotite",
        # Other (12)
        "gypsum", "halite", "fluorite", "barite", "apatite",
        "talc", "graphite", "sulfur", "diamond", "sylvite",
        "kaolinite", "serpentine"
    ],

    "coffee variety": [
        # Species (10)
        "arabica", "robusta", "liberica", "excelsa", "eugenioides",
        "stenophylla", "mauritiana", "racemosa", "congensis", "canephora",
        # Arabica cultivars (20)
        "bourbon", "typica", "gesha", "caturra", "catuai",
        "mundo novo", "pacamara", "maragogype", "sl28", "sl34",
        "kent", "java", "blue mountain", "kona", "santos",
        "sidra", "pacas", "villa sarchi", "tekisic", "maracaturra",
        # Origins & Processing (20)
        "ethiopian heirloom", "colombian", "brazilian", "costa rican", "guatemalan",
        "kenyan", "tanzanian", "yemeni", "sumatran", "hawaiian",
        "jamaican", "mexican", "peruvian", "honduran", "salvadoran",
        "washed", "natural", "honey process", "wet hulled", "anaerobic"
    ],

    "telescope type": [
        # Optical designs (15)
        "refractor", "reflector", "catadioptric", "newtonian", "cassegrain",
        "schmidt cassegrain", "maksutov", "dobsonian", "ritchey chretien", "gregorian",
        "herschelian", "nasmyth", "coude", "off axis", "prime focus",
        # By wavelength (12)
        "optical", "radio", "infrared", "ultraviolet", "x-ray",
        "gamma ray", "submillimeter", "microwave", "far infrared", "near infrared",
        "extreme ultraviolet", "soft x-ray",
        # Mounting & Special (18)
        "equatorial", "altazimuth", "german equatorial", "fork mount", "transit",
        "zenith", "solar", "lunar", "planetary", "deep sky",
        "wide field", "astrograph", "coronagraph", "spectrograph", "photometer",
        "space telescope", "ground based", "interferometer"
    ],

    "martial art": [
        # East Asian (18)
        "karate", "judo", "taekwondo", "kung fu", "aikido",
        "jiu jitsu", "kendo", "iaido", "ninjutsu", "hapkido",
        "wing chun", "tai chi", "shaolin", "wushu", "bajiquan",
        "xingyiquan", "baguazhang", "shuai jiao",
        # Southeast Asian (10)
        "muay thai", "silat", "kali", "escrima", "arnis",
        "pencak silat", "krabi krabong", "tomoi", "bokator", "pradal serey",
        # Brazilian & South American (5)
        "brazilian jiu jitsu", "capoeira", "vale tudo", "luta livre", "huka huka",
        # Western (12)
        "boxing", "wrestling", "kickboxing", "savate", "pankration",
        "catch wrestling", "greco roman", "freestyle wrestling", "sambo", "systema",
        "bartitsu", "combatives",
        # Other (10)
        "krav maga", "mixed martial arts", "jeet kune do", "submission wrestling", "shootfighting",
        "shooto", "pancrase", "combat sambo", "sanda", "kyokushin"
    ],

    "sea creature": [
        # Marine mammals (15)
        "dolphin", "whale", "orca", "porpoise", "manatee",
        "sea lion", "seal", "walrus", "sea otter", "narwhal",
        "beluga", "sperm whale", "humpback whale", "blue whale", "gray whale",
        # Fish (15)
        "shark", "ray", "tuna", "swordfish", "marlin",
        "barracuda", "eel", "grouper", "snapper", "sea bass",
        "flounder", "halibut", "mahi mahi", "sailfish", "wahoo",
        # Invertebrates (15)
        "octopus", "squid", "cuttlefish", "nautilus", "jellyfish",
        "sea anemone", "coral", "sponge", "sea cucumber", "sea urchin",
        "starfish", "brittle star", "sea lily", "sand dollar", "sea fan",
        # Crustaceans & Others (10)
        "crab", "lobster", "shrimp", "krill", "barnacle",
        "sea turtle", "seahorse", "sea dragon", "mantis shrimp", "hermit crab"
    ],

    "psychology term": [
        # Cognitive (15)
        "cognition", "perception", "attention", "memory", "learning",
        "thinking", "reasoning", "problem solving", "decision making", "language",
        "intelligence", "consciousness", "metacognition", "executive function", "working memory",
        # Behavioral (12)
        "behavior", "conditioning", "reinforcement", "punishment", "extinction",
        "habituation", "sensitization", "shaping", "chaining", "schedules",
        "operant", "classical",
        # Emotional & Social (13)
        "emotion", "motivation", "personality", "attitude", "attribution",
        "conformity", "obedience", "persuasion", "prejudice", "stereotype",
        "aggression", "altruism", "empathy",
        # Clinical & Development (15)
        "schema", "heuristic", "bias", "cognitive dissonance", "defense mechanism",
        "attachment", "identity", "socialization", "moral development", "ego",
        "superego", "id", "unconscious", "repression", "transference"
    ],

    "chemical element": [
        # Common elements (20)
        "hydrogen", "helium", "carbon", "nitrogen", "oxygen",
        "fluorine", "neon", "sodium", "magnesium", "aluminum",
        "silicon", "phosphorus", "sulfur", "chlorine", "argon",
        "potassium", "calcium", "iron", "copper", "zinc",
        # Metals (15)
        "lithium", "beryllium", "titanium", "chromium", "manganese",
        "nickel", "silver", "gold", "platinum", "mercury",
        "lead", "tin", "tungsten", "uranium", "plutonium",
        # Noble gases & Halogens (10)
        "krypton", "xenon", "radon", "bromine", "iodine",
        "astatine", "helium gas", "neon light", "argon gas", "hydrogen gas"
    ],

    "dinosaur genus": [
        # Theropods (18)
        "tyrannosaurus", "velociraptor", "allosaurus", "spinosaurus", "carnotaurus",
        "giganotosaurus", "carcharodontosaurus", "albertosaurus", "deinonychus", "utahraptor",
        "compsognathus", "gallimimus", "ornithomimus", "therizinosaurus", "oviraptor",
        "coelophysis", "dilophosaurus", "megalosaurus",
        # Sauropods (12)
        "brachiosaurus", "diplodocus", "apatosaurus", "brontosaurus", "argentinosaurus",
        "titanosaurus", "sauroposeidon", "camarasaurus", "mamenchisaurus", "supersaurus",
        "seismosaurus", "alamosaurus",
        # Ornithischians (15)
        "triceratops", "stegosaurus", "ankylosaurus", "iguanodon", "parasaurolophus",
        "pachycephalosaurus", "styracosaurus", "protoceratops", "pentaceratops", "corythosaurus",
        "edmontosaurus", "maiasaura", "psittacosaurus", "hypsilophodon", "stegoceras"
    ],

    "programming language": [
        # Modern high-level (15)
        "python", "javascript", "java", "c++", "c#",
        "ruby", "go", "rust", "swift", "kotlin",
        "typescript", "php", "scala", "r", "matlab",
        # Functional & Academic (10)
        "haskell", "ocaml", "lisp", "scheme", "clojure",
        "erlang", "elixir", "f#", "ml", "prolog",
        # Systems & Low-level (10)
        "c", "assembly", "fortran", "cobol", "ada",
        "pascal", "d", "zig", "nim", "crystal",
        # Scripting & Others (15)
        "perl", "bash", "powershell", "lua", "tcl",
        "awk", "sed", "groovy", "julia", "dart",
        "objective-c", "visual basic", "delphi", "scratch", "labview"
    ],

    "ancient civilization": [
        # Mediterranean (15)
        "egyptian", "greek", "roman", "phoenician", "minoan",
        "mycenaean", "etruscan", "carthaginian", "hittite", "lydian",
        "thracian", "illyrian", "macedonian", "spartan", "athenian",
        # Mesopotamian (10)
        "sumerian", "babylonian", "assyrian", "akkadian", "persian",
        "median", "elamite", "chaldean", "amorite", "hurrian",
        # American (12)
        "aztec", "mayan", "incan", "olmec", "toltec",
        "zapotec", "teotihuacan", "mixtec", "hohokam", "anasazi",
        "mississippian", "moche",
        # Asian & Other (13)
        "chinese", "indus valley", "harappan", "mohenjo daro", "khmer",
        "nubian", "kushite", "axumite", "sabaean", "nabataean",
        "parthian", "scythian", "celt"
    ],

    "bridge type": [
        # Basic types (15)
        "beam", "arch", "suspension", "cable stayed", "truss",
        "cantilever", "tied arch", "movable", "swing", "drawbridge",
        "bascule", "lift", "transporter", "pontoon", "floating",
        # Arch variations (10)
        "stone arch", "concrete arch", "steel arch", "through arch", "deck arch",
        "corbel arch", "pointed arch", "segmental arch", "elliptical arch", "parabolic arch",
        # Truss types (15)
        "warren truss", "pratt truss", "howe truss", "lattice truss", "k truss",
        "baltimore truss", "pennsylvania truss", "camelback truss", "bowstring truss", "deck truss",
        "through truss", "pony truss", "quadrangular truss", "space frame", "tubular truss",
        # Special (10)
        "covered", "pedestrian", "railway", "aqueduct", "viaduct",
        "trestle", "culvert", "skyway", "footbridge", "overpass"
    ],

    "photography technique": [
        # Camera techniques (15)
        "long exposure", "short exposure", "panning", "tracking", "zooming",
        "bracketing", "hdr", "focus stacking", "time lapse", "hyperlapse",
        "slow motion", "high speed", "bulb mode", "multiple exposure", "double exposure",
        # Composition (12)
        "rule of thirds", "golden ratio", "leading lines", "framing", "symmetry",
        "patterns", "negative space", "depth of field", "bokeh", "selective focus",
        "foreground interest", "layering",
        # Genres (13)
        "portrait", "landscape", "macro", "street photography", "wildlife",
        "astrophotography", "architectural", "documentary", "fashion", "sports",
        "aerial", "underwater", "food photography",
        # Post-processing (10)
        "black and white", "sepia", "cross processing", "tilt shift", "vignetting",
        "light painting", "silhouette", "contre jour", "golden hour", "blue hour"
    ],

    "boat type": [
        # Sailing (15)
        "sailboat", "yacht", "sloop", "ketch", "schooner",
        "catamaran", "trimaran", "dinghy", "yawl", "cutter",
        "brig", "brigantine", "barque", "clipper", "junk",
        # Motor (15)
        "speedboat", "motorboat", "cabin cruiser", "trawler", "tugboat",
        "ferry", "hydrofoil", "hovercraft", "jetski", "pontoon",
        "bowrider", "deck boat", "cuddy cabin", "center console", "walkaround",
        # Commercial & Special (15)
        "cargo ship", "container ship", "tanker", "cruise ship", "ocean liner",
        "fishing boat", "trawler boat", "seiner", "dragger", "longliner",
        "icebreaker", "research vessel", "submarine", "lifeboat", "pilot boat",
        # Small craft (10)
        "canoe", "kayak", "rowboat", "dory", "skiff",
        "gondola", "coracle", "raft", "inflatable", "houseboat"
    ],

    "cartoon character": [
        # Classic Disney (12)
        "mickey mouse", "donald duck", "goofy", "pluto", "minnie mouse",
        "daisy duck", "chip and dale", "clarabelle cow", "pete", "huey dewey louie",
        "scrooge mcduck", "gyro gearloose",
        # Warner Bros (12)
        "bugs bunny", "daffy duck", "porky pig", "elmer fudd", "tweety",
        "sylvester", "yosemite sam", "foghorn leghorn", "marvin martian", "road runner",
        "wile e coyote", "tasmanian devil",
        # Hanna-Barbera (10)
        "fred flintstone", "barney rubble", "yogi bear", "huckleberry hound", "quick draw mcgraw",
        "snagglepuss", "scooby doo", "tom and jerry", "top cat", "jetsons",
        # Modern (16)
        "spongebob", "patrick star", "homer simpson", "bart simpson", "peter griffin",
        "stewie griffin", "eric cartman", "kenny mccormick", "beavis", "butthead",
        "ren and stimpy", "powerpuff girls", "dexter", "johnny bravo", "courage",
        "ed edd eddy"
    ],

    "stadium name": [
        # European football (15)
        "wembley", "camp nou", "old trafford", "bernabeu", "san siro",
        "allianz arena", "signal iduna", "emirates", "anfield", "etihad",
        "stamford bridge", "parc des princes", "olympiastadion", "estadio da luz", "celtic park",
        # American sports (15)
        "yankee stadium", "fenway park", "wrigley field", "dodger stadium", "lambeau field",
        "cowboys stadium", "soldier field", "arrowhead", "gillette stadium", "metlife stadium",
        "lucas oil", "us bank", "sofi stadium", "allegiant stadium", "state farm stadium",
        # Multi-purpose & International (20)
        "madison square garden", "united center", "staples center", "td garden", "wells fargo center",
        "rose bowl", "cotton bowl", "sugar bowl", "orange bowl", "fiesta bowl",
        "maracana", "azteca", "la bombonera", "melbourne cricket ground", "sydney cricket ground",
        "eden gardens", "lords", "twickenham", "millennium stadium", "aviva stadium"
    ],

    "surgical procedure": [
        # General (15)
        "appendectomy", "cholecystectomy", "hernia repair", "mastectomy", "hysterectomy",
        "colectomy", "gastrectomy", "splenectomy", "nephrectomy", "prostatectomy",
        "thyroidectomy", "parathyroidectomy", "adrenalectomy", "pancreaticoduodenectomy", "lobectomy",
        # Orthopedic (12)
        "hip replacement", "knee replacement", "shoulder replacement", "spinal fusion", "laminectomy",
        "meniscectomy", "arthroscopy", "rotator cuff repair", "acl reconstruction", "carpal tunnel release",
        "bunionectomy", "hip resurfacing",
        # Cardiac & Vascular (10)
        "bypass surgery", "valve replacement", "angioplasty", "stenting", "pacemaker insertion",
        "defibrillator implant", "endarterectomy", "aneurysm repair", "varicose vein surgery", "thrombectomy",
        # Other (13)
        "cataract surgery", "lasik", "tonsillectomy", "adenoidectomy", "cesarean section",
        "transplant", "biopsy", "laparoscopy", "endoscopy", "colonoscopy",
        "rhinoplasty", "facelift", "liposuction"
    ],

    "constellation": [
        # Zodiac (12)
        "aries", "taurus", "gemini", "cancer", "leo",
        "virgo", "libra", "scorpius", "sagittarius", "capricornus",
        "aquarius", "pisces",
        # Northern (18)
        "ursa major", "ursa minor", "cassiopeia", "cepheus", "draco",
        "cygnus", "lyra", "aquila", "andromeda", "perseus",
        "auriga", "bootes", "corona borealis", "hercules", "pegasus",
        "delphinus", "lacerta", "camelopardalis",
        # Southern (20)
        "orion", "canis major", "canis minor", "centaurus", "crux",
        "carina", "vela", "puppis", "hydra", "corvus",
        "crater", "lupus", "ara", "corona australis", "pavo",
        "tucana", "phoenix", "grus", "indus", "octans"
    ],

    "spice blend": [
        # Regional blends (20)
        "curry powder", "garam masala", "tandoori masala", "chaat masala", "panch phoron",
        "ras el hanout", "berbere", "baharat", "za'atar", "dukkah",
        "harissa", "chermoula", "shichimi togarashi", "furikake", "chinese five spice",
        "herbes de provence", "fines herbes", "bouquet garni", "cajun seasoning", "creole seasoning",
        # American & European (15)
        "italian seasoning", "greek seasoning", "taco seasoning", "fajita seasoning", "chili powder",
        "pumpkin pie spice", "apple pie spice", "poultry seasoning", "steak seasoning", "blackening seasoning",
        "old bay", "jerk seasoning", "adobo", "sazon", "sofrito",
        # Other (15)
        "curry paste", "masala paste", "mole", "recado", "achiote",
        "sambal", "nam prik", "gochugaru", "togarashi", "wasabi",
        "mustard blend", "pickling spice", "mulling spice", "fish spice", "seafood seasoning"
    ],

    "guitar type": [
        # Acoustic (15)
        "acoustic", "classical", "flamenco", "steel string", "twelve string",
        "dreadnought", "jumbo", "parlor", "grand concert", "auditorium",
        "grand auditorium", "orchestra model", "slope shoulder", "archtop acoustic", "resonator",
        # Electric solid body (12)
        "electric", "stratocaster", "telecaster", "les paul", "sg",
        "flying v", "explorer", "firebird", "jaguar", "jazzmaster",
        "superstrat", "offset",
        # Electric hollow (8)
        "semi hollow", "hollow body", "archtop electric", "thinline", "es 335",
        "gretsch", "rickenbacker", "gibson es",
        # Specialty (15)
        "bass", "baritone", "seven string", "eight string", "lap steel",
        "pedal steel", "dobro", "mandolin guitar", "travel guitar", "silent guitar",
        "parlor guitar", "tenor guitar", "jazz guitar", "surf guitar", "slide guitar"
    ],

    "hat style": [
        # Formal (12)
        "fedora", "trilby", "homburg", "pork pie", "bowler",
        "top hat", "derby", "panama", "boater", "straw boater",
        "gambler", "plantation",
        # Casual (15)
        "baseball cap", "snapback", "trucker hat", "dad hat", "fitted cap",
        "visor", "bucket hat", "boonie hat", "fisherman hat", "crusher",
        "newsboy cap", "flat cap", "ivy cap", "driving cap", "gatsby",
        # Winter (10)
        "beanie", "watch cap", "ski cap", "knit cap", "stocking cap",
        "trapper hat", "ushanka", "bomber", "ear flap", "balaclava",
        # Specialty (13)
        "beret", "tam", "cowboy hat", "stetson", "sombrero",
        "fez", "kufi", "turban", "keffiyeh", "toque",
        "cloche", "fascinator", "picture hat"
    ],

    "painting medium": [
        # Traditional (15)
        "oil", "acrylic", "watercolor", "gouache", "tempera",
        "fresco", "encaustic", "casein", "enamel", "alkyd",
        "egg tempera", "distemper", "grisaille", "verdaccio", "impasto",
        # Drawing (12)
        "pastel", "chalk", "charcoal", "conte crayon", "graphite",
        "colored pencil", "ink", "marker", "pen and ink", "brush and ink",
        "sanguine", "sepia",
        # Modern (13)
        "spray paint", "airbrush", "digital", "mixed media", "collage",
        "assemblage", "photomontage", "screen print", "block print", "lithograph",
        "etching", "engraving", "woodcut",
        # Specialty (10)
        "miniature", "illumination", "icon", "mural", "trompe l'oeil",
        "gesso", "primer", "varnish", "glaze", "wash"
    ],

    "volcano name": [
        # Active major (15)
        "kilauea", "etna", "stromboli", "vesuvius", "krakatoa",
        "mount fuji", "mount st helens", "pinatubo", "eyjafjallajokull", "popocatepetl",
        "cotopaxi", "mauna loa", "nyiragongo", "merapi", "sakurajima",
        # Historic eruptions (15)
        "tambora", "santorini", "thera", "mount pelee", "nevado del ruiz",
        "unzen", "galunggung", "kelud", "mount lamington", "arenal",
        "paricutin", "surtsey", "heimaey", "mount agung", "taal",
        # Famous peaks (20)
        "mount rainier", "mount shasta", "mount hood", "kilimanjaro", "mount kenya",
        "elbrus", "ararat", "damavand", "kazbek", "pico de orizaba",
        "chimborazo", "tungurahua", "villarrica", "llaima", "osorno",
        "ruapehu", "tongariro", "taranaki", "krakatau", "anak krakatau"
    ],

    "fruit variety": [
        # Citrus (10)
        "orange", "lemon", "lime", "grapefruit", "tangerine",
        "mandarin", "clementine", "pomelo", "kumquat", "bergamot",
        # Berries (12)
        "strawberry", "blueberry", "raspberry", "blackberry", "cranberry",
        "gooseberry", "currant", "elderberry", "boysenberry", "loganberry",
        "mulberry", "huckleberry",
        # Stone fruits (10)
        "peach", "plum", "cherry", "apricot", "nectarine",
        "prune", "damson", "greengage", "mirabelle", "sloe",
        # Tropical (15)
        "banana", "mango", "pineapple", "papaya", "guava",
        "passionfruit", "dragonfruit", "lychee", "rambutan", "mangosteen",
        "durian", "jackfruit", "starfruit", "longan", "persimmon",
        # Other (13)
        "apple", "pear", "grape", "watermelon", "cantaloupe",
        "honeydew", "kiwi", "fig", "date", "pomegranate",
        "quince", "medlar", "loquat"
    ],

    "sword type": [
        # European (18)
        "longsword", "broadsword", "rapier", "saber", "claymore",
        "zweihander", "greatsword", "bastard sword", "arming sword", "short sword",
        "gladius", "spatha", "falchion", "cutlass", "epee",
        "foil", "smallsword", "estoc",
        # Asian (15)
        "katana", "wakizashi", "tanto", "nodachi", "tachi",
        "dao", "jian", "dadao", "butterfly sword", "hook sword",
        "kris", "kampilan", "talwar", "khanda", "shamshir",
        # Middle Eastern & Other (17)
        "scimitar", "kilij", "yataghan", "khanjar", "khopesh",
        "shotel", "nimcha", "saif", "takoba", "ida",
        "makhaira", "kopis", "xiphos", "parazonium", "pugio",
        "seax", "scramasax"
    ],

    "board game": [
        # Abstract strategy (15)
        "chess", "checkers", "go", "backgammon", "othello",
        "mancala", "nine mens morris", "tic tac toe", "connect four", "mastermind",
        "pente", "gomoku", "shogi", "xiangqi", "stratego",
        # Euro games (15)
        "catan", "carcassonne", "ticket to ride", "agricola", "puerto rico",
        "power grid", "el grande", "tzolkin", "terra mystica", "brass",
        "caverna", "concordia", "scythe", "wingspan", "azul",
        # American (12)
        "monopoly", "risk", "clue", "scrabble", "trivial pursuit",
        "life", "sorry", "battleship", "operation", "mousetrap",
        "chutes and ladders", "candy land",
        # Modern (8)
        "pandemic", "betrayal", "arkham horror", "gloomhaven", "descent",
        "twilight imperium", "dominant species", "food chain magnate"
    ]
}


def generate_interleaved_dataset(
    num_categories: int = 46,
    interference_levels: List[int] = [3, 10, 50, 100, 200, 300, 400, 500],
    output_path: str = "data/interleaved_dataset_meaningful.json"
):
    """
    Generate interleaved dataset with expanded meaningful real-world values.

    Args:
        num_categories: Number of categories to track (default 46)
        interference_levels: List of update counts per category
        output_path: Where to save the JSON file
    """

    categories = list(CATEGORY_VALUES.keys())[:num_categories]

    dataset = {
        "metadata": {
            "description": "Interleaved dataset for retroactive interference experiments with EXPANDED meaningful values (50-70+ per category)",
            "num_categories": num_categories,
            "categories": categories,
            "interference_levels": interference_levels,
            "value_type": "expanded_meaningful_real_world",
            "generated": "2025-12-27",
            "note": "interference_level = total appearances (baseline + interference). Level 3 = 1 baseline + 2 interference updates."
        },
        "levels": {}
    }

    for level in interference_levels:
        print(f"Generating level {level}...")

        # Sample values for this level (baseline + updates)
        level_values = {}
        for category in categories:
            available_values = CATEGORY_VALUES[category].copy()
            random.shuffle(available_values)

            # Need level unique values (1 baseline + (level-1) updates = level total)
            needed = level
            category_values = available_values[:min(needed, len(available_values))]

            # If we need more values than available, create variants
            if len(category_values) < needed:
                base_values = available_values
                for i in range(needed - len(category_values)):
                    variant_idx = i % len(base_values)
                    category_values.append(f"{base_values[variant_idx]} {i+1}")

            level_values[category] = category_values

        # Generate all updates (WITHOUT marking baselines yet)
        all_updates = []
        for cat in categories:
            for update_idx in range(level):
                all_updates.append({
                    "category": cat,
                    "value": level_values[cat][update_idx],
                    "update_index": update_idx
                })

        # Shuffle to interleave
        random.shuffle(all_updates)

        # NOW track baseline positions and values (first occurrence of each category)
        baseline_positions = {}
        baseline_values = {}
        sequence = []
        seen_categories = set()

        for idx, update in enumerate(all_updates):
            cat = update["category"]
            val = update["value"]

            # First occurrence = baseline
            if cat not in seen_categories:
                baseline_positions[cat] = idx
                baseline_values[cat] = val
                seen_categories.add(cat)

            sequence.append({
                "category": cat,
                "value": val
            })

        # Store level data
        dataset["levels"][str(level)] = {
            "sequence_length": len(sequence),
            "updates_per_category": level,
            "baseline_values": baseline_values,
            "baseline_positions": baseline_positions,
            "interleaved_sequence": sequence
        }

    # Save to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"\n✅ Dataset generated successfully!")
    print(f"   Output: {output_path}")
    print(f"   Categories: {num_categories}")
    print(f"   Levels: {interference_levels}")
    print(f"   Total experiments: {num_categories * len(interference_levels)}")

    # Print value counts
    print(f"\n📊 Value counts per category:")
    for cat in categories[:10]:
        print(f"   {cat}: {len(CATEGORY_VALUES[cat])} values")
    print("   ...")

    return dataset


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)

    # Generate dataset
    dataset = generate_interleaved_dataset(
        num_categories=46,
        interference_levels=[3, 10, 50, 100, 200, 300, 400, 500],
        output_path="data/interleaved_dataset_meaningful.json"
    )

    print("\n✅ Done! Ready to run experiments with EXPANDED meaningful values (50-70+ per category).")
