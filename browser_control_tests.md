Lancer et répondre à un quizz : 

uv run cli agents langchain -p "Browser Agent Direct" -m haiku@edenai "va sur jetpunk et répond au quiz du jour"

Remplir un google form :

uv run cli agents langchain -p "Browser Agent Direct" -m haiku@edenai "répond à ce formulaire : https://docs.google.com/forms/d/e/1FAIpQLSesQK7fQEg6R0OrutoQanOviVf2oU1O7Sbzv0D_vc_dNx23pQ/viewform?usp=publish-editor. Fait attention au question a choix multiple, tu peut cocher plusieurs cases. Lis la page une seul fois et répond aux question les unes à la suite des autres (avec quelques seconde d'écart entre chaque réponse) puis valide le formulaire"

Tenter de remplir une note de frais (le cadre "note de frais" doit être dans la page d'acceuil myatos) : 

uv run cli agents langchain -p "Browser Agent Direct" -m haiku@edenai "Utilise tes skills à disposition pour aller sur la page de connection de myatos, attendre que l'utilisateur se connecte et cliquer sur note de frais. Créer une note de frais, remplis la première étape avec les valeur par défaut du skill dédié, puis passe à la suivante"