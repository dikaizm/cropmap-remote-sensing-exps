# CDL class IDs that are NOT crops (non-agricultural land cover).
# Everything not in this set (and not 0/background) is treated as a potential crop class.
CDL_NON_CROP_IDS: frozenset = frozenset({
    0,              # Background / NoData
    61,             # Fallow/Idle Cropland  → remapped to background (class 0)
    63, 64, 65,     # Forest / Shrubland / Barren
    81,             # Clouds/No Data
    82,             # Developed
    83,             # Water
    87, 88,         # Wetlands / Nonag/Undefined
    92,             # Aquaculture
    111, 112,       # Open Water / Perennial Ice/Snow
    121, 122, 123, 124,  # Developed variants
    131,            # Barren
    141, 142, 143,  # Forest variants
    152,            # Shrubland
    176,            # Grass/Pasture
    190, 195,       # Woody/Herbaceous Wetlands
})

USDA_CDL_COLORS = {
    0:  "#000000",  # Background
    1:  "#ffd400",  # Corn
    2:  "#ff2626",  # Cotton
    3:  "#00a9e6",  # Rice
    4:  "#ff9e0f",  # Sorghum
    5:  "#267300",  # Soybeans
    6:  "#ffff00",  # Sunflower
    10: "#70a800",  # Peanuts
    11: "#00af4d",  # Tobacco
    12: "#e0a60f",  # Sweet Corn
    13: "#e0a60f",  # Pop or Orn Corn
    14: "#80d4ff",  # Mint
    21: "#e2007f",  # Barley
    22: "#8a6453",  # Durum Wheat
    23: "#d9b56c",  # Spring Wheat
    24: "#a87000",  # Winter Wheat
    25: "#d69dbc",  # Other Small Grains
    26: "#737300",  # Dbl Crop WinWht/Soybeans
    27: "#ae017e",  # Rye
    28: "#a15889",  # Oats
    29: "#73004c",  # Millet
    30: "#d69dbc",  # Speltz
    31: "#d1ff00",  # Canola
    32: "#8099ff",  # Flaxseed
    33: "#d6d600",  # Safflower
    34: "#d1ff00",  # Rape Seed
    35: "#00af4d",  # Mustard
    36: "#ffa8e3",  # Alfalfa
    37: "#a5f58d",  # Other Hay/Non Alfalfa
    38: "#00af4d",  # Camelina
    39: "#d69dbc",  # Buckwheat
    41: "#a900e6",  # Sugarbeets
    42: "#a80000",  # Dry Beans
    43: "#732600",  # Potatoes
    44: "#00af4d",  # Other Crops
    45: "#b380ff",  # Sugarcane
    46: "#732600",  # Sweet Potatoes
    47: "#ff6666",  # Misc Vegs & Fruits
    48: "#ff6666",  # Watermelons
    49: "#ffcc66",  # Onions
    50: "#ff6666",  # Cucumbers
    51: "#00af4d",  # Chick Peas
    52: "#00deb0",  # Lentils
    53: "#55ff00",  # Peas
    54: "#f5a27a",  # Tomatoes
    55: "#ff6666",  # Caneberries
    56: "#00af4d",  # Hops
    57: "#80d4ff",  # Herbs
    58: "#e8beff",  # Clover/Wildflowers
    59: "#b2ffde",  # Sod/Grass Seed
    60: "#00af4d",  # Switchgrass
    61: "#bfbf7a",  # Fallow/Idle Cropland
    63: "#95ce93",  # Forest
    64: "#c7d79e",  # Shrubland
    65: "#ccbfa3",  # Barren
    66: "#ff00ff",  # Cherries
    67: "#ff91ab",  # Peaches
    68: "#b90050",  # Apples
    69: "#704489",  # Grapes
    70: "#007878",  # Christmas Trees
    71: "#b39c70",  # Other Tree Crops
    72: "#ffff80",  # Citrus
    74: "#b6705c",  # Pecans
    75: "#00a884",  # Almonds
    76: "#ebd6b0",  # Walnuts
    77: "#b39c70",  # Pears
    81: "#f7f7f7",  # Clouds/No Data
    82: "#9c9c9c",  # Developed
    83: "#4d70a3",  # Water
    87: "#80b3b3",  # Wetlands
    88: "#e9ffbe",  # Nonag/Undefined
    92: "#00ffff",  # Aquaculture
    111: "#4d70a3", # Open Water
    112: "#d4e3fc", # Perennial Ice/Snow
    121: "#9c9c9c", # Developed/Open Space
    122: "#9c9c9c", # Developed/Low Intensity
    123: "#9c9c9c", # Developed/Med Intensity
    124: "#9c9c9c", # Developed/High Intensity
    131: "#ccbfa3", # Barren
    141: "#95ce93", # Deciduous Forest
    142: "#95ce93", # Evergreen Forest
    143: "#95ce93", # Mixed Forest
    152: "#c7d79e", # Shrubland
    176: "#e9ffbe", # Grass/Pasture
    190: "#80b3b3", # Woody Wetlands
    195: "#80b3b3", # Herbaceous Wetlands
    204: "#00ff8c", # Pistachios
    205: "#d69dbc", # Triticale
    206: "#ff6666", # Carrots
    207: "#ff6666", # Asparagus
    208: "#ff6666", # Garlic
    209: "#ff6666", # Cantaloupes
    210: "#ff91ab", # Prunes
    211: "#344a34", # Olives
    212: "#e67525", # Oranges
    213: "#ff6666", # Honeydew Melons
    214: "#ff6666", # Broccoli
    215: "#66994d", # Avocados
    216: "#ff6666", # Peppers
    217: "#b39c70", # Pomegranates
    218: "#ff91ab", # Nectarines
    219: "#ff6666", # Greens
    220: "#ff91ab", # Plums
    221: "#ff6666", # Strawberries
    222: "#ff6666", # Squash
    223: "#ff91ab", # Apricots
    224: "#00af4d", # Vetch
    225: "#ffd400", # Dbl Crop WinWht/Corn
    226: "#ffd400", # Dbl Crop Oats/Corn
    227: "#ff6666", # Lettuce
    228: "#ffd400", # Dbl Crop Triticale/Corn
    229: "#ff6666", # Pumpkins
    230: "#8a6453", # Dbl Crop Lettuce/Durum Wht
    231: "#ff6666", # Dbl Crop Lettuce/Cantaloupe
    232: "#ff2626", # Dbl Crop Lettuce/Cotton
    233: "#e2007f", # Dbl Crop Lettuce/Barley
    234: "#ff9e0f", # Dbl Crop Durum Wht/Sorghum
    235: "#ff9e0f", # Dbl Crop Barley/Sorghum
    236: "#a87000", # Dbl Crop WinWht/Sorghum
    237: "#ffd400", # Dbl Crop Barley/Corn
    238: "#a87000", # Dbl Crop WinWht/Cotton
    239: "#267300", # Dbl Crop Soybeans/Cotton
    240: "#267300", # Dbl Crop Soybeans/Oats
    241: "#ffd400", # Dbl Crop Corn/Soybeans
    242: "#000099", # Blueberries
    243: "#ff6666", # Cabbage
    244: "#ff6666", # Cauliflower
    245: "#ff6666", # Celery
    246: "#ff6666", # Radishes
    247: "#ff6666", # Turnips
    248: "#ff6666", # Eggplants
    249: "#ff6666", # Gourds
    250: "#ff6666", # Cranberries
    254: "#267300", # Dbl Crop Barley/Soybeans
}

USDA_CDL_NAMES = {
    0:  "Background",
    1:  "Corn",
    2:  "Cotton",
    3:  "Rice",
    4:  "Sorghum",
    5:  "Soybeans",
    6:  "Sunflower",
    10: "Peanuts",
    11: "Tobacco",
    12: "Sweet Corn",
    13: "Pop or Orn Corn",
    14: "Mint",
    21: "Barley",
    22: "Durum Wheat",
    23: "Spring Wheat",
    24: "Winter Wheat",
    25: "Other Small Grains",
    26: "Dbl Crop WinWht/Soybeans",
    27: "Rye",
    28: "Oats",
    29: "Millet",
    30: "Speltz",
    31: "Canola",
    32: "Flaxseed",
    33: "Safflower",
    34: "Rape Seed",
    35: "Mustard",
    36: "Alfalfa",
    37: "Other Hay/Non Alfalfa",
    38: "Camelina",
    39: "Buckwheat",
    41: "Sugarbeets",
    42: "Dry Beans",
    43: "Potatoes",
    44: "Other Crops",
    45: "Sugarcane",
    46: "Sweet Potatoes",
    47: "Misc Vegs & Fruits",
    48: "Watermelons",
    49: "Onions",
    50: "Cucumbers",
    51: "Chick Peas",
    52: "Lentils",
    53: "Peas",
    54: "Tomatoes",
    55: "Caneberries",
    56: "Hops",
    57: "Herbs",
    58: "Clover/Wildflowers",
    59: "Sod/Grass Seed",
    60: "Switchgrass",
    61: "Fallow/Idle Cropland",
    63: "Forest",
    64: "Shrubland",
    65: "Barren",
    66: "Cherries",
    67: "Peaches",
    68: "Apples",
    69: "Grapes",
    70: "Christmas Trees",
    71: "Other Tree Crops",
    72: "Citrus",
    74: "Pecans",
    75: "Almonds",
    76: "Walnuts",
    77: "Pears",
    81: "Clouds/No Data",
    82: "Developed",
    83: "Water",
    87: "Wetlands",
    88: "Nonag/Undefined",
    92: "Aquaculture",
    111: "Open Water",
    112: "Perennial Ice/Snow",
    121: "Developed/Open Space",
    122: "Developed/Low Intensity",
    123: "Developed/Med Intensity",
    124: "Developed/High Intensity",
    131: "Barren",
    141: "Deciduous Forest",
    142: "Evergreen Forest",
    143: "Mixed Forest",
    152: "Shrubland",
    176: "Grass/Pasture",
    190: "Woody Wetlands",
    195: "Herbaceous Wetlands",
    204: "Pistachios",
    205: "Triticale",
    206: "Carrots",
    207: "Asparagus",
    208: "Garlic",
    209: "Cantaloupes",
    210: "Prunes",
    211: "Olives",
    212: "Oranges",
    213: "Honeydew Melons",
    214: "Broccoli",
    215: "Avocados",
    216: "Peppers",
    217: "Pomegranates",
    218: "Nectarines",
    219: "Greens",
    220: "Plums",
    221: "Strawberries",
    222: "Squash",
    223: "Apricots",
    224: "Vetch",
    225: "Dbl Crop WinWht/Corn",
    226: "Dbl Crop Oats/Corn",
    227: "Lettuce",
    228: "Dbl Crop Triticale/Corn",
    229: "Pumpkins",
    230: "Dbl Crop Lettuce/Durum Wht",
    231: "Dbl Crop Lettuce/Cantaloupe",
    232: "Dbl Crop Lettuce/Cotton",
    233: "Dbl Crop Lettuce/Barley",
    234: "Dbl Crop Durum Wht/Sorghum",
    235: "Dbl Crop Barley/Sorghum",
    236: "Dbl Crop WinWht/Sorghum",
    237: "Dbl Crop Barley/Corn",
    238: "Dbl Crop WinWht/Cotton",
    239: "Dbl Crop Soybeans/Cotton",
    240: "Dbl Crop Soybeans/Oats",
    241: "Dbl Crop Corn/Soybeans",
    242: "Blueberries",
    243: "Cabbage",
    244: "Cauliflower",
    245: "Celery",
    246: "Radishes",
    247: "Turnips",
    248: "Eggplants",
    249: "Gourds",
    250: "Cranberries",
    254: "Dbl Crop Barley/Soybeans",
}