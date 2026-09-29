"""Static normalisation dictionaries (general language knowledge, no external data).

Country-specific because abbreviations collide across regions: "st" is
"street" in the US but "saint" in France; "or" is Odisha in India but an
English word elsewhere.
"""

# --- Business names ---------------------------------------------------------
LEGAL = {  # token -> canonical legal form
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "llc": "llc", "llp": "llp", "lp": "lp",
    "ltd": "ltd", "limited": "ltd", "pvt": "pvt", "private": "pvt",
    "public": "public", "plc": "plc", "pllc": "pllc", "pc": "pc", "opc": "opc",
    # France
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "sci": "sci",
    "eurl": "eurl", "snc": "snc", "selarl": "selarl",
}
NAME_STOP = {"the", "and", "of", "a", "an", "du", "de", "des", "la", "le", "les",
             "et", "l", "d", "en", "aux", "au", "m", "s", "ms"}
DBA = r"\b(doing business as|d ?b ?a|trading as|t ?a|also known as|aka|formerly)\b"

# --- Addresses --------------------------------------------------------------
ADDR_COMMON = {
    "no": "", "number": "", "num": "", "door": "", "hno": "", "h": "",
    "the": "", "of": "", "and": "",
}
ADDR_ABBR = {
    "US": {
        "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue",
        "blvd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court", "hwy": "highway",
        "pl": "place", "sq": "square", "ste": "suite", "fl": "floor", "flr": "floor",
        "apt": "apartment", "bldg": "building", "pkwy": "parkway", "cir": "circle",
        "ter": "terrace", "trl": "trail", "hts": "heights", "mt": "mount", "ft": "fort",
        "pt": "point", "rte": "route", "fwy": "freeway", "expy": "expressway",
        "ctr": "center", "plz": "plaza", "cv": "cove", "xing": "crossing", "pk": "park",
        "n": "north", "s": "south", "so": "south", "e": "east", "w": "west",
        "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
        "po": "po", "box": "box", "unit": "", "suite": "suite",
    },
    "India": {
        "rd": "road", "st": "street", "nr": "near", "opp": "opposite", "mkt": "market",
        "ngr": "nagar", "sec": "sector", "ph": "phase", "extn": "extension", "ext": "extension",
        "blk": "block", "hsg": "housing", "soc": "society", "indl": "industrial",
        "est": "estate", "dist": "district", "bldg": "building", "apt": "apartment",
        "fl": "floor", "flr": "floor", "cplx": "complex", "chwk": "chowk", "tq": "taluk",
        "tal": "taluk", "po": "post", "ps": "police",
    },
    "France": {
        "r": "rue", "av": "avenue", "ave": "avenue", "bd": "boulevard", "bld": "boulevard",
        "boul": "boulevard", "pl": "place", "imp": "impasse", "rte": "route", "ch": "chemin",
        "che": "chemin", "all": "allee", "sq": "square", "crs": "cours", "fbg": "faubourg",
        "qu": "quai", "res": "residence", "resid": "residence", "st": "saint", "ste": "sainte",
        "gen": "general", "gal": "general", "mal": "marechal", "pres": "president",
        "bis": "bis", "ter": "ter", "apt": "appartement", "app": "appartement",
        "bat": "batiment", "de": "", "du": "", "des": "", "la": "", "le": "", "les": "",
        "l": "", "d": "", "chez": "", "mme": "", "mr": "", "m": "",
    },
}

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont",
    "va": "virginia", "wa": "washington", "wv": "west virginia", "wi": "wisconsin",
    "wy": "wyoming", "dc": "district of columbia", "pr": "puerto rico",
}
INDIA_STATES = {
    "ap": "andhra pradesh", "ar": "arunachal pradesh", "as": "assam", "br": "bihar",
    "cg": "chhattisgarh", "ga": "goa", "gj": "gujarat", "hr": "haryana",
    "hp": "himachal pradesh", "jh": "jharkhand", "ka": "karnataka", "kl": "kerala",
    "mp": "madhya pradesh", "mh": "maharashtra", "mn": "manipur", "ml": "meghalaya",
    "mz": "mizoram", "nl": "nagaland", "od": "odisha", "pb": "punjab", "rj": "rajasthan",
    "sk": "sikkim", "tn": "tamil nadu", "ts": "telangana", "tr": "tripura",
    "up": "uttar pradesh", "uk": "uttarakhand", "wb": "west bengal",
    "an": "andaman and nicobar islands", "ch": "chandigarh", "dl": "delhi",
    "jk": "jammu and kashmir", "la": "ladakh", "ld": "lakshadweep", "py": "puducherry",
    "dn": "dadra and nagar haveli and daman and diu",
}
INDIA_STATE_ALIASES = {
    "orissa": "od", "or": "od", "tg": "ts", "ct": "cg", "chattisgarh": "cg",
    "ut": "uk", "uttaranchal": "uk", "pondicherry": "py", "new delhi": "dl",
    "nct of delhi": "dl", "delhi ncr": "dl", "j&k": "jk", "jammu & kashmir": "jk",
}
CITY_ALIASES = {
    "India": {"bangalore": "bengaluru", "bombay": "mumbai", "greater bombay": "mumbai",
              "madras": "chennai", "calcutta": "kolkata", "gurgaon": "gurugram",
              "poona": "pune", "baroda": "vadodara", "trivandrum": "thiruvananthapuram",
              "mysore": "mysuru", "mangalore": "mangaluru", "calicut": "kozhikode",
              "cochin": "kochi", "benares": "varanasi", "allahabad": "prayagraj"},
}


# France: sources name either the region ("Hauts-de-France") or the departement ("Nord"),
# so both map to the region code. Keys are lowercase, accents already stripped.
FRANCE_REGIONS = {
    "HDF": ["hauts-de-france", "hauts de france", "nord", "pas-de-calais", "pas de calais", "aisne", "oise", "somme"],
    "NAQ": ["nouvelle-aquitaine", "nouvelle aquitaine", "gironde", "landes", "dordogne", "lot-et-garonne",
            "pyrenees-atlantiques", "charente", "charente-maritime", "correze", "creuse", "deux-sevres",
            "vienne", "haute-vienne"],
    "PDL": ["pays de la loire", "pays-de-la-loire", "loire-atlantique", "loire atlantique", "maine-et-loire",
            "mayenne", "sarthe", "vendee"],
    "IDF": ["ile-de-france", "ile de france", "paris", "seine-saint-denis", "hauts-de-seine", "val-de-marne",
            "essonne", "yvelines", "val-d'oise", "seine-et-marne"],
    "ARA": ["auvergne-rhone-alpes", "rhone"], "BFC": ["bourgogne-franche-comte"], "BRE": ["bretagne"],
    "CVL": ["centre-val de loire"], "GES": ["grand est"], "NOR": ["normandie"], "OCC": ["occitanie"],
    "PAC": ["provence-alpes-cote dazur", "provence-alpes-cote d'azur", "bouches-du-rhone"], "COR": ["corse"],
}


def state_map(country):
    """Address component (lowercase) -> canonical state code, for one country."""
    if country == "US":
        m = {k: k for k in US_STATES}
        m.update({v: k for k, v in US_STATES.items()})
        return m
    if country == "India":
        m = {k: k for k in INDIA_STATES}
        m.update({v: k for k, v in INDIA_STATES.items()})
        m.update(INDIA_STATE_ALIASES)
        return m
    if country == "France":
        return {name: code for code, names in FRANCE_REGIONS.items() for name in names}
    return {}
