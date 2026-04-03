import genanki
import os

# 1. Define your Model (The fields on your card)
my_model = genanki.Model(
  None,
  'English-Portugues 3000 Organizado',
  fields=[
    {'name': 'Question'},
    {'name': 'Answer'},
    {'name': 'MyImage'}, # Dedicated field for the image tag
  ],
  templates=[{
    'name': 'Card 1',
    'qfmt': '{{Question}}',
    'afmt': '{{FrontSide}}<hr id="answer">{{Answer}}<br>{{MyImage}}',
  }])

# 2. Create the Deck object
my_deck = genanki.Deck(2059400110, 'My Programmatic Deck')
