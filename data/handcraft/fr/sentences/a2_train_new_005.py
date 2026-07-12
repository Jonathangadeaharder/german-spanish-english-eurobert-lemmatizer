"""100 handcrafted French A2 sentences for a2_train_new_005 (fr_a2_train_801–900)."""

# 4 batches × 25 sentences = 100 (A2: 5–12 tokens, passé composé / futur proche)
# Topics: camping | French lessons | art museum | community help

BATCH_1 = [
    # 801–825: camping
    "Nous avons monté la tente près du lac.",
    "Je vais allumer un feu de camp.",
    "Elle a préparé le dîner dehors.",
    "Ils ont dormi sous les étoiles.",
    "Tu vas apporter ton sac de couchage ?",
    "J'ai trouvé un bon emplacement.",
    "Elle a entendu des oiseaux la nuit.",
    "Nous avons marché jusqu'à la rivière.",
    "Il va ranger la tente demain matin.",
    "Je vais cuisiner des saucisses au feu.",
    "Elle a pris des photos du coucher de soleil.",
    "Nous allons rester deux nuits ici.",
    "Il a mis ses chaussures de randonnée.",
    "Tu as déjà campé dans cette forêt ?",
    "J'ai emporté une lampe de poche.",
    "Elle a partagé son goûter avec nous.",
    "Nous avons écouté le vent dans les arbres.",
    "Il va chercher du bois sec.",
    "Je vais mettre ma veste chaude.",
    "Elle a trouvé un sentier balisé.",
    "Nous allons partir avant la pluie.",
    "Il a raconté des histoires au feu.",
    "Tu vas dormir dans la tente avec nous ?",
    "Je vais laver la vaisselle à la rivière.",
    "Elle a aimé la nuit au camping.",
]

BATCH_2 = [
    # 826–850: French lessons
    "J'ai suivi un cours de français hier.",
    "Elle va réviser les verbes irréguliers.",
    "Nous avons écouté un enregistrement.",
    "Il a appris dix mots nouveaux.",
    "Tu vas faire les exercices ce soir ?",
    "Je vais lire un texte en français.",
    "Elle a posé une question au professeur.",
    "Nous avons pratiqué la prononciation.",
    "Il va préparer un petit exposé.",
    "J'ai écrit une phrase au tableau.",
    "Elle a corrigé mes fautes d'orthographe.",
    "Nous allons regarder un film en français.",
    "Il a répété la leçon à voix haute.",
    "Tu as déjà parlé avec un francophone ?",
    "Je vais apprendre une nouvelle expression.",
    "Elle a trouvé un bon dictionnaire.",
    "Nous avons joué à un jeu de vocabulaire.",
    "Il va écouter une chanson française.",
    "Je vais noter les mots difficiles.",
    "Elle a progressé depuis le mois dernier.",
    "Nous allons travailler en petits groupes.",
    "Il a lu un article dans le journal.",
    "Tu vas réviser pour le test ?",
    "Je vais demander de l'aide au professeur.",
    "Elle a aimé la leçon d'aujourd'hui.",
]

BATCH_3 = [
    # 851–875: art museum
    "Nous avons visité le musée d'art.",
    "Je vais acheter un billet d'entrée.",
    "Elle a admiré un tableau célèbre.",
    "Ils ont pris un audioguide.",
    "Tu vas venir avec nous samedi ?",
    "J'ai regardé une sculpture moderne.",
    "Elle a lu la description de l'œuvre.",
    "Nous avons fait une visite guidée.",
    "Il a pris des notes sur le peintre.",
    "Je vais retrouver mes amis à l'entrée.",
    "Elle a trouvé l'exposition très intéressante.",
    "Nous allons voir une autre salle.",
    "Il a acheté une carte postale.",
    "Tu as déjà vu ce tableau en photo ?",
    "J'ai reconnu un artiste français.",
    "Elle a parlé avec le guide.",
    "Nous avons regardé des dessins anciens.",
    "Il va dessiner dans son carnet.",
    "Je vais attendre devant la boutique.",
    "Elle a aimé les couleurs du tableau.",
    "Nous allons revenir pour une autre exposition.",
    "Il a trouvé le musée très calme.",
    "Tu vas prendre des photos sans flash ?",
    "Je vais noter le nom de l'artiste.",
    "Elle a appris l'histoire du musée.",
]

BATCH_4 = [
    # 876–900: community help
    "J'ai aidé à la bibliothèque du quartier.",
    "Elle va participer à une collecte de vivres.",
    "Nous avons nettoyé le parc communal.",
    "Il a donné des vêtements à l'association.",
    "Tu vas venir à la réunion citoyenne ?",
    "Je vais distribuer des flyers.",
    "Elle a préparé des repas pour les sans-abri.",
    "Nous avons planté des arbres ensemble.",
    "Il va aider les personnes âgées.",
    "J'ai écrit une annonce pour le local.",
    "Elle a accueilli de nouveaux bénévoles.",
    "Nous allons organiser une fête de quartier.",
    "Il a réparé un banc du parc.",
    "Tu as déjà fait du bénévolat ici ?",
    "Je vais apporter des fournitures scolaires.",
    "Elle a parlé avec les voisins.",
    "Nous avons collecté des jouets pour enfants.",
    "Il va préparer le café pour l'événement.",
    "Je vais inviter plus de monde.",
    "Elle a trouvé le travail très utile.",
    "Nous allons continuer l'aide chaque mois.",
    "Il a remercié tous les bénévoles.",
    "Tu vas rejoindre notre groupe ?",
    "Je vais proposer une nouvelle idée.",
    "Elle a aimé aider sa communauté.",
]

BATCHES = [BATCH_1, BATCH_2, BATCH_3, BATCH_4]
SENTENCES = [s for batch in BATCHES for s in batch]

assert len(SENTENCES) == 100, f"Expected 100 sentences, got {len(SENTENCES)}"
for i, batch in enumerate(BATCHES, 1):
    assert len(batch) == 25, f"BATCH_{i} has {len(batch)} sentences"