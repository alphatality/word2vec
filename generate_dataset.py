import random

articles = ["the", "this", "that", "every"]

subject_adjectives = ["brave", "robotic", "modern", "alien", "curious", "smart", "fast"]
subject_nouns = ["astronaut", "spaceship", "rover", "satellite", "probe", "ship", "crew", "telescope"]

adverbs = ["carefully", "silently", "quickly", "slowly", "accurately", "eagerly", "boldly"]

verbs = [
    "observes", "explores", "orbits", "discovers", 
    "scans", "approaches", "studies", "maps",
    "photographs", "analyzes"
]

object_adjectives = [
    "distant", "massive", "mysterious", "unknown", 
    "bright", "rocky", "frozen", "ancient",
    "glowing", "hostile", "dark"
]

object_nouns = [
    "planet", "galaxy", "star", "asteroid", 
    "moon", "nebula", "comet", "system",
    "meteor", "pulsar"
]

prepositions = ["near", "around", "beyond", "towards", "outside", "above"]
place_nouns = ["orbit", "space", "void", "cosmos", "horizon", "station"]

def generate_sentences(num_sentences):
    sentences = set()
    
    while len(sentences) < num_sentences:
        template = random.randint(1, 3)
        
        art1 = random.choice(articles)
        subj_adj = random.choice(subject_adjectives)
        subj_noun = random.choice(subject_nouns)
        adv = random.choice(adverbs)
        verb = random.choice(verbs)
        art2 = random.choice(articles)
        obj_adj = random.choice(object_adjectives)
        obj_noun = random.choice(object_nouns)
        prep = random.choice(prepositions)
        art3 = random.choice(articles)
        place = random.choice(place_nouns)
        
        if template == 1:
            sentence = f"{art1} {subj_adj} {subj_noun} {adv} {verb} {art2} {obj_adj} {obj_noun}"
        elif template == 2:
            sentence = f"{art1} {subj_noun} {verb} {art2} {obj_adj} {obj_noun} {prep} {art3} {obj_adj} {place}"
        else:
            sentence = f"{art1} {subj_adj} {subj_noun} {adv} {verb} {art2} {obj_adj} {obj_noun} {prep} {art3} {place}"
            
        sentences.add(sentence.capitalize())
        
    return list(sentences)

def main():
    train_data = generate_sentences(1000)
    test_data = generate_sentences(200)

    with open("data/train.txt", "w", encoding="utf-8") as f:
        for line in train_data:
            f.write(line + "\n")
            
    with open("data/test.txt", "w", encoding="utf-8") as f:
        for line in test_data:
            f.write(line + "\n")


main()