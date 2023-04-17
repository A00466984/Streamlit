import pandas as pd
import sqlite3

con = sqlite3.connect("data/Chinook_Sqlite.sqlite")
cursor = con.cursor()

query = "Select * from artist"
artists = pd.DataFrame(cursor.execute(query).fetchall(), columns=['artist_id', 'artist_name'])
query = "Select * from album"
albums = pd.DataFrame(cursor.execute(query).fetchall(), columns=['album_id', 'album_title', 'artist_id'])
query = "Select ar.ArtistId, ar.Name, al.Title from album al inner join artist ar on ar.ArtistId = al.ArtistId where ar.Name = \'AC/DC\'"
result = cursor.execute(query).fetchall()
artistalbum = albums.set_index('artist_id').join(artists, on=['artist_id'])


