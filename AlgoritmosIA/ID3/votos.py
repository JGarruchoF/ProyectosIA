# ========================================================
# Conjunto de datos "Congressional Voting Record"
# Fuente: UCI Machine Learning Repository 
# (http://archive.ics.uci.edu/ml/)
# ========================================================


# Atributo de clasificación
# -------------------------

atributo_clasificación='Partido'


# Valores de clasificación:
# -------------------------

clases=['republicano','demócrata']


# Atributos (o características):
# ------------------------------

# Se trata de 16 votaciones realizadas a lo largo del año 1984:

# 1. handicapped-infants
# 2. water-project-cost-sharing
# 3. adoption-of-the-budget-resolution
# 4. physician-fee-freeze
# 5. el-salvador-aid
# 6. religious-groups-in-schools
# 7. anti-satellite-test-ban
# 8. aid-to-nicaraguan-contras
# 9.  mx-missile
# 10. immigration
# 11. synfuels-corporation-cutback
# 12. education-spending
# 13. superfund-right-to-sue
# 14. crime
# 15. duty-free-exports
# 16. export-administration-act-south-africa

# Por simplificar, los atributos se denominarán "votox" con "x" de 1 a 16 

# Los valores posibles de cada voto son sí ("s"), no ("no") y presente
# ("?"). El voto "presente", en el congreso de Estados Unidos, podríamos
# asimilarlo al voto en blanco (aunque no es exactamente lo mismo); en
# cualquier caso, no lo trataremos como un "valor desconocido", sino como un
# tercer tipo de votación, que no es ni "si" ni "no".  

atributos=[('voto1',["s","n","?"]),
           ('voto2',["s","n","?"]),
           ('voto3',["s","n","?"]),
           ('voto4',["s","n","?"]),
           ('voto5',["s","n","?"]),
           ('voto6',["s","n","?"]),
           ("voto7",["s","n","?"]),
           ('voto8',["s","n","?"]),
           ('voto9',["s","n","?"]),
           ('voto10',["s","n","?"]),
           ('voto11',["s","n","?"]),
           ('voto12',["s","n","?"]),
           ('voto13',["s","n","?"]),
           ('voto14',["s","n","?"]),
           ('voto15',["s","n","?"]),
           ('voto16',["s","n","?"])]



# Ejemplos del conjunto de entrenamiento:
# ---------------------------------------


entr=[['n','s','n','s','s','s','n','n','n','s','?','s','s','s','n','s','republicano'],
      ['n','s','n','s','s','s','n','n','n','n','n','s','s','s','n','?','republicano'],
      ['?','s','s','?','s','s','n','n','n','n','s','n','s','s','n','n','demócrata' ],
      ['n','s','s','n','?','s','n','n','n','n','s','n','s','n','n','s','demócrata' ],
      ['s','s','s','n','s','s','n','n','n','n','s','?','s','s','s','s','demócrata' ],
      ['n','s','s','n','s','s','n','n','n','n','n','n','s','s','s','s','demócrata' ],
      ['n','s','n','s','s','s','n','n','n','n','n','n','?','s','s','s','demócrata' ],
      ['n','s','n','s','s','s','n','n','n','n','n','n','s','s','?','s','republicano'],
      ['n','s','n','s','s','s','n','n','n','n','n','s','s','s','n','s','republicano'],
      ['s','s','s','n','n','n','s','s','s','n','n','n','n','n','?','?','demócrata' ],
      ['n','s','n','s','s','n','n','n','n','n','?','?','s','s','n','n','republicano'],
      ['n','s','n','s','s','s','n','n','n','n','s','?','s','s','?','?','republicano'],
      ['n','s','s','n','n','n','s','s','s','n','n','n','s','n','?','?','demócrata' ],
      ['s','s','s','n','n','s','s','s','?','s','s','?','n','n','s','?','demócrata' ],
      ['n','s','n','s','s','s','n','n','n','n','n','s','?','?','n','?','republicano'],
      ['n','s','n','s','s','s','n','n','n','s','n','s','s','?','n','?','republicano'],
      ['s','n','s','n','n','s','n','s','?','s','s','s','?','n','n','s','demócrata' ],
      ['s','?','s','n','n','n','s','s','s','n','n','n','s','n','s','s','demócrata' ],
      ['n','s','n','s','s','s','n','n','n','n','n','?','s','s','n','n','republicano'],
      ['s','s','s','n','n','n','s','s','s','n','s','n','n','n','s','s','demócrata' ],
      ['s','s','s','n','n','?','s','s','n','n','s','n','n','n','s','s','demócrata' ],
      ['s','s','s','n','n','n','s','s','s','n','n','n','?','?','s','s','demócrata' ],
      ['s','?','s','n','n','n','s','s','s','n','n','?','n','n','s','s','demócrata' ],
      ['s','s','s','n','n','n','s','s','s','n','n','n','n','n','s','s','demócrata' ],
      ['s','n','s','n','n','n','s','s','s','n','n','n','n','n','s','?','demócrata' ],
      ['s','n','s','n','n','n','s','s','s','s','n','n','n','n','s','s','demócrata' ],
      ['s','n','s','n','n','n','s','s','s','n','s','n','n','n','s','s','demócrata' ],
      ['s','s','s','n','n','n','s','s','s','n','s','n','n','n','s','s','demócrata' ],
      ['s','n','n','s','s','n','s','s','s','n','n','s','s','s','n','s','republicano'],
      ['s','s','s','n','n','n','s','s','s','n','s','n','n','n','s','s','demócrata' ],
      ['n','s','n','s','s','s','n','n','n','n','n','s','s','s','n','n','republicano'],
      ['s','s','s','n','n','n','s','s','s','n','s','n','n','n','s','?','demócrata' ],
      ['s','s','s','n','n','n','s','s','s','s','n','n','s','n','s','s','demócrata' ],
      ['n','s','n','s','s','s','n','n','n','n','n','s','s','s','n','s','republicano'],
      ['s','s','s','n','n','n','s','s','s','n','n','n','n','n','s','s','demócrata' ],
      ['n','s','n','s','s','s','n','n','n','n','n','s','s','s','n','n','republicano'],
      ['s','?','n','s','s','s','n','n','n','s','n','s','?','s','n','s','republicano'],
      ['s','s','n','s','s','s','n','n','n','n','n','n','s','s','n','s','republicano'],
      ['n','s','n','s','s','s','n','n','n','s','n','s','s','s','n','n','republicano'],
      ['s','n','s','n','n','n','s','s','s','s','s','n','s','n','s','s','demócrata' ],
      ['s','s','s','n','n','n','s','s','s','n','?','n','n','n','n','?','demócrata' ],
      ['s','s','s','n','n','n','s','s','s','n','n','n','n','n','s','?','demócrata' ],
      ['s','n','s','n','n','n','s','s','s','n','n','n','n','n','n','s','demócrata' ],
      ['s','n','s','n','n','n','s','s','s','n','n','n','n','n','s','s','demócrata' ],
      ['s','s','s','n','n','n','s','s','s','n','s','n','n','n','n','?','demócrata' ],
      ['s','s','s','n','n','n','s','s','?','n','s','n','n','n','s','?','demócrata' ],
      ['s','s','s','n','n','n','s','s','s','n','n','n','n','n','n','s','demócrata' ],
      ['s','n','s','n','n','n','s','s','?','n','n','n','n','n','n','?','demócrata' ],
      ['s','s','s','n','n','n','s','s','n','n','n','n','n','s','n','s','demócrata' ],
      ['n','?','n','s','s','s','n','n','n','n','n','s','s','s','n','n','republicano'],
      ['s','s','s','n','n','n','s','s','s','n','s','n','n','n','s','s','demócrata' ],
      ['n','s','n','s','s','s','n','?','n','n','n','s','s','s','n','s','republicano'],
      ['s','s','s','n','n','n','s','s','s','n','n','n','n','n','?','?','demócrata' ],
      ['s','s','n','s','s','s','n','n','n','s','n','s','s','s','n','n','republicano'],
      ['s','s','s','n','n','s','?','s','n','n','s','s','n','s','n','?','demócrata' ],
      ['n','s','n','s','s','s','n','n','n','s','s','s','s','s','n','n','republicano'],
      ['n','s','n','s','s','s','n','n','n','s','s','s','s','s','n','s','republicano'],
      ['n','s','n','s','s','s','n','n','n','s','n','s','s','s','n','s','republicano'],
      ['n','s','n','s','s','s','n','n','n','s','n','s','s','s','n','s','republicano'],
      ['n','s','n','s','s','s','n','n','n','s','n','s','s','s','n','?','republicano'],
      ['s','s','s','n','n','?','s','s','s','s','n','n','n','n','s','?','demócrata' ],
      ['n','s','n','s','s','s','n','n','n','n','n','s','s','s','n','n','republicano'],
      ['s','s','s','n','n','n','s','s','s','n','n','n','n','n','n','?','demócrata' ],
      ['s','s','s','n','n','n','s','s','s','n','s','n','n','n','n','s','demócrata' ],
      ['s','s','s','n','n','n','s','s','s','n','s','?','n','n','n','s','demócrata' ],
      ['s','s','n','s','s','s','s','n','n','n','n','s','s','s','n','s','republicano'],
      ['n','s','n','s','s','s','s','n','n','n','s','s','s','s','n','s','republicano'],
      ['n','s','n','s','s','s','n','n','n','s','n','s','s','s','n','n','republicano'],
      ['s','?','s','n','n','n','s','s','s','n','n','n','s','n','s','s','demócrata' ],
      ['s','s','s','n','n','n','s','s','s','n','n','n','n','n','s','s','demócrata' ],
      ['s','n','s','n','n','n','s','s','s','n','n','n','s','n','s','?','demócrata' ],
      ['s','s','s','s','n','n','s','s','s','s','s','n','n','s','n','s','republicano'],
      ['s','s','s','n','n','n','s','s','s','n','s','n','n','n','s','?','demócrata' ],
      ['s','n','s','s','s','n','s','n','s','s','n','n','s','s','n','s','republicano'],
      ['s','n','s','n','n','s','s','s','s','s','s','n','n','s','s','s','demócrata' ],
      ['n','s','s','s','s','s','n','n','n','s','s','n','s','s','n','n','demócrata' ],
      ['n','s','s','n','s','s','n','n','n','s','s','s','s','s','n','?','demócrata' ],
      ['n','s','s','s','s','s','n','s','s','s','s','s','s','s','n','s','demócrata' ],
      ['s','s','s','n','s','s','n','n','n','s','s','n','s','s','n','s','demócrata' ],
      ['n','n','n','s','s','n','n','n','n','s','n','s','s','s','n','n','republicano'],
      ['s','n','s','n','n','s','s','s','s','s','n','s','n','s','n','?','demócrata' ],
      ['s','n','s','n','n','n','s','s','?','s','s','s','n','s','n','s','demócrata' ],
      ['n','n','n','s','s','s','n','n','n','s','n','s','s','s','n','s','republicano'],
      ['n','n','n','s','s','s','n','n','n','n','n','s','s','s','n','n','republicano'],
      ['n','?','n','s','s','s','n','n','n','s','n','s','s','s','n','n','republicano'],
      ['n','n','s','n','s','s','n','n','n','s','s','s','s','s','n','s','demócrata' ],
      ['n','n','n','s','s','s','n','n','n','s','n','s','s','s','n','n','republicano'],
      ['n','n','n','s','s','s','n','n','n','n','n','s','s','s','n','n','republicano'],
      ['n','s','s','n','s','s','s','n','s','s','s','n','s','s','n','s','demócrata' ],
      ['n','n','n','s','s','s','n','n','n','s','n','?','s','s','n','?','republicano'],
      ['s','n','s','n','n','n','s','s','s','s','n','n','n','n','s','s','demócrata' ],
      ['s','n','s','n','n','n','s','s','s','s','s','n','n','n','s','s','demócrata' ],
      ['s','s','s','n','n','n','s','s','n','s','s','n','n','?','s','s','demócrata' ],
      ['s','n','s','n','n','n','s','n','s','s','s','n','n','n','s','s','demócrata' ],
      ['s','n','s','n','s','s','n','n','n','n','n','n','n','n','n','s','demócrata' ],
      ['s','n','s','n','s','s','n','?','?','n','s','?','?','?','s','s','demócrata' ],
      ['n','n','?','n','s','s','n','n','n','n','s','s','s','s','n','s','demócrata' ],
      ['s','n','n','n','s','s','s','n','n','s','s','n','n','s','n','s','demócrata' ],
      ['s','s','s','n','n','s','s','s','s','s','n','n','n','n','n','s','demócrata' ],
      ['n','n','n','s','s','s','n','n','n','s','?','s','s','s','n','n','republicano'],
      ['s','n','n','n','s','s','n','n','n','n','s','s','n','s','n','s','demócrata' ],
      ['s','n','s','n','s','s','s','n','n','n','s','n','n','s','n','s','demócrata' ],
      ['s','n','s','n','s','s','s','n','?','n','s','n','s','s','s','?','demócrata' ],
      ['s','n','n','n','s','s','?','n','?','n','n','n','n','s','?','n','demócrata' ],
      ['?','?','?','?','n','s','s','s','s','s','?','n','s','s','n','?','demócrata' ],
      ['s','s','s','n','n','n','n','s','s','n','s','n','n','n','s','s','demócrata' ],
      ['n','s','n','s','s','s','n','n','n','n','n','s','s','s','n','s','republicano'],
      ['n','?','?','?','?','?','?','?','?','?','?','?','?','s','?','?','republicano'],
      ['s','?','s','n','n','n','s','s','s','n','n','n','n','n','s','?','demócrata' ],
      ['s','?','s','n','n','n','s','s','s','n','n','n','n','n','s','?','demócrata' ],
      ['n','n','s','n','n','n','s','s','s','s','n','n','n','n','s','s','demócrata' ],
      ['n','?','n','s','s','s','n','n','n','s','n','s','s','s','n','s','republicano'],
      ['n','?','s','n','n','s','s','s','n','s','n','n','n','n','s','?','demócrata' ],
      ['n','?','n','s','s','s','n','n','n','s','n','s','s','s','n','n','republicano'],
      ['s','?','s','n','n','n','s','s','s','n','n','n','n','n','s','?','demócrata' ],
      ['n','?','s','n','?','?','s','s','s','s','?','?','n','n','s','s','demócrata' ],
      ['s','n','s','n','n','n','s','s','s','n','s','n','n','n','s','s','demócrata' ],
      ['s','s','s','s','s','n','s','n','n','n','n','s','s','s','n','s','republicano'],
      ['n','s','s','n','n','n','n','s','s','s','s','n','n','n','s','s','demócrata' ],
      ['n','n','n','s','s','s','n','n','n','n','n','s','s','s','n','n','republicano'],
      ['n','?','?','s','s','s','n','n','n','s','n','s','s','s','?','s','republicano'],
      ['n','?','n','s','s','s','n','n','n','s','n','s','s','s','n','s','republicano'],
      ['n','n','n','s','s','s','n','n','n','s','n','s','n','s','n','s','republicano'],
      ['s','?','n','s','s','s','n','s','n','n','n','s','s','s','n','s','republicano'],
      ['n','?','s','n','n','n','s','s','s','n','n','n','n','n','s','s','demócrata' ],
      ['n','?','n','s','s','s','n','n','n','s','n','s','s','s','n','s','republicano'],
      ['n','?','n','s','s','s','n','n','n','n','n','s','s','s','n','n','republicano'],
      ['n','?','s','n','n','n','s','s','s','s','s','n','n','s','s','s','demócrata' ],
      ['n','?','s','n','n','s','n','s','n','s','s','n','n','n','s','s','demócrata' ],
      ['?','?','s','n','n','n','s','s','?','n','?','?','?','?','?','?','demócrata' ],
      ['s','?','s','n','?','?','s','s','s','n','n','n','n','n','s','?','demócrata' ],
      ['n','n','s','n','n','s','n','s','s','s','n','n','n','s','n','s','demócrata' ],
      ['n','n','n','s','s','s','n','n','n','s','n','s','s','s','n','?','republicano'],
      ['n','n','n','s','s','s','n','n','n','s','n','s','s','s','n','s','republicano'],
      ['n','n','n','s','s','s','n','n','n','n','n','s','s','s','n','?','republicano'],
      ['n','n','n','s','s','s','n','n','n','s','n','s','s','s','n','n','republicano'],
      ['n','s','n','s','s','s','n','n','n','s','s','s','s','n','n','s','republicano'],
      ['n','?','s','n','n','s','s','s','s','s','n','n','n','s','s','s','demócrata' ],
      ['n','n','s','n','n','s','s','s','s','s','n','n','n','s','n','s','demócrata' ],
      ['s','n','s','n','n','s','s','s','s','n','n','n','n','n','s','s','demócrata' ],
      ['n','n','n','s','n','n','s','s','s','s','n','n','s','s','n','s','republicano'],
      ['n','n','n','s','s','s','s','s','s','s','n','s','s','s','?','s','republicano'],
      ['n','n','n','s','s','s','s','s','s','s','n','s','s','s','n','s','republicano'],
      ['?','s','n','n','n','n','s','s','s','s','s','n','n','s','s','s','demócrata' ],
      ['n','?','n','n','n','s','s','s','s','s','n','n','n','s','n','?','demócrata' ],
      ['n','n','s','n','n','s','s','s','s','s','n','n','n','s','?','s','demócrata' ],
      ['n','s','n','s','s','s','n','n','n','n','n','s','s','s','n','s','republicano'],
      ['n','n','n','n','n','n','s','s','s','s','n','s','s','s','s','s','demócrata' ],
      ['n','s','n','s','s','s','n','n','n','s','s','s','s','s','n','s','republicano'],
      ['n','n','s','n','n','n','s','s','s','s','n','n','s','n','s','s','demócrata' ],
      ['s','s','n','s','s','s','n','n','n','s','n','s','s','s','n','s','republicano'],
      ['s','s','?','s','s','s','n','n','s','n','s','?','s','s','n','n','demócrata' ],
      ['n','s','s','n','n','s','n','s','s','s','s','n','s','n','s','s','demócrata' ],
      ['n','n','s','n','n','s','s','s','s','s','s','n','s','s','n','s','demócrata' ],
      ['n','s','n','s','s','s','n','n','n','n','n','s','s','s','n','n','republicano'],
      ['s','s','n','s','s','s','n','?','n','n','s','s','s','s','n','n','republicano'],
      ['s','s','n','s','s','s','s','n','n','n','n','s','s','s','n','n','republicano'],
      ['n','s','s','n','n','s','n','s','s','n','s','n','?','?','?','?','demócrata' ],
      ['n','s','n','s','s','s','n','n','n','s','n','s','s','s','n','n','republicano'],
      ['n','s','s','n','?','s','s','s','s','s','s','n','n','?','n','?','demócrata' ],
      ['n','s','n','n','s','s','n','n','n','n','n','s','s','s','s','s','demócrata' ],
      ['n','n','n','n','s','s','s','n','n','n','n','s','s','s','n','s','demócrata' ],
      ['n','s','s','n','s','s','s','n','n','n','s','s','s','s','n','s','demócrata' ],
      ['n','s','n','s','s','s','s','n','n','n','n','s','s','s','n','s','republicano'],
      ['s','s','n','n','s','s','n','n','n','s','s','s','s','s','n','?','demócrata' ],
      ['n','s','s','n','n','s','s','s','s','s','s','n','s','n','s','?','demócrata' ],
      ['s','n','s','s','s','s','s','s','n','s','n','s','n','s','s','s','republicano'],
      ['s','n','s','s','s','s','s','s','n','s','s','s','n','s','s','s','republicano'],
      ['n','n','s','s','s','s','n','n','s','n','n','n','s','s','s','?','demócrata' ],
      ['s','n','s','n','n','n','s','s','s','s','s','n','n','s','n','s','demócrata' ],
      ['s','n','s','n','n','n','?','s','s','?','n','n','n','n','s','?','demócrata' ],
      ['n','?','n','s','s','s','n','n','n','s','n','s','s','s','n','s','republicano'],
      ['n','s','s','n','n','n','s','s','s','s','n','n','?','n','s','s','demócrata' ],
      ['n','n','n','n','s','s','n','n','n','s','s','s','s','s','n','s','demócrata' ],
      ['s','?','s','n','n','n','s','s','s','n','n','n','n','n','s','?','demócrata' ],
      ['n','s','s','n','n','n','s','s','s','s','n','n','n','n','s','s','demócrata' ],
      ['n','n','s','s','n','n','s','s','s','s','n','n','n','s','s','s','republicano'],
      ['n','n','s','n','n','n','s','s','s','s','s','?','n','n','s','s','demócrata' ],
      ['?','n','s','n','n','n','s','s','s','s','s','?','n','n','s','?','demócrata' ],
      ['s','n','s','n','n','n','s','s','s','s','n','n','n','n','s','s','demócrata' ],
      ['?','?','s','n','n','n','s','s','s','?','?','n','n','n','?','?','demócrata' ],
      ['n','n','s','n','n','n','s','s','s','s','s','n','n','n','s','s','demócrata' ],
      ['s','?','s','n','n','n','s','s','s','n','n','n','n','n','s','s','demócrata' ],
      ['?','?','?','?','?','?','?','?','s','?','?','?','?','?','?','?','demócrata' ],
      ['n','n','s','n','n','n','s','s','s','s','s','n','n','n','s','s','demócrata' ],
      ['s','n','s','n','n','n','s','s','s','s','n','?','n','n','s','s','demócrata' ],
      ['n','s','s','n','n','n','s','s','s','s','s','n','n','n','s','s','demócrata' ],
      ['s','n','s','n','n','n','s','s','s','n','n','n','n','n','s','?','demócrata' ],
      ['s','?','n','s','s','s','s','s','n','n','n','s','?','s','?','?','republicano'],
      ['s','n','s','n','n','n','s','s','s','n','n','n','n','n','s','s','demócrata' ],
      ['n','?','n','s','s','s','n','n','n','n','n','s','s','s','n','?','republicano'],
      ['n','s','n','s','s','s','n','?','n','s','n','s','s','s','n','?','republicano'],
      ['n','n','n','n','n','s','s','s','s','n','s','n','n','s','s','s','demócrata' ],
      ['n','n','s','n','n','n','s','s','s','n','n','n','n','n','s','s','demócrata' ],
      ['n','n','s','n','n','s','s','?','s','s','s','n','n','n','s','s','demócrata' ],
      ['n','n','n','s','s','s','n','n','n','n','n','s','s','s','n','?','republicano'],
      ['n','n','s','n','n','s','s','s','s','n','s','s','n','s','s','?','demócrata' ],
      ['n','?','s','s','s','s','n','n','n','s','n','n','n','s','n','s','republicano'],
      ['n','n','s','n','n','n','s','s','s','s','s','n','?','n','s','?','demócrata' ],
      ['s','s','n','n','n','n','s','s','?','n','s','n','n','n','s','?','demócrata' ],
      ['n','n','s','n','n','n','s','s','s','n','n','n','n','s','s','s','demócrata' ],
      ['s','s','s','n','n','n','s','s','s','n','n','n','n','n','s','s','demócrata' ],
      ['s','n','s','n','n','s','s','s','s','s','s','n','n','n','s','s','demócrata' ],
      ['s','n','s','n','n','n','s','s','s','s','n','n','n','n','s','s','demócrata' ],
      ['n','n','s','s','s','s','s','n','n','n','n','s','s','s','n','s','republicano'],
      ['n','n','s','n','n','s','s','s','s','s','n','s','n','n','n','s','demócrata' ],
      ['n','n','n','s','s','s','n','n','n','s','n','s','n','s','n','s','republicano'],
      ['s','?','n','s','s','s','s','n','n','s','n','s','s','s','n','s','republicano'],
      ['n','n','s','n','n','n','s','s','s','n','n','?','n','n','s','s','demócrata' ],
      ['s','s','s','n','n','n','s','s','s','s','s','n','n','n','n','s','demócrata' ],
      ['n','n','s','n','n','s','s','s','s','n','n','n','n','n','s','s','demócrata' ],
      ['n','s','n','s','s','s','n','n','n','s','n','s','s','s','n','s','republicano'],
      ['n','n','s','n','n','n','s','s','s','n','s','n','n','n','s','s','demócrata' ],
      ['n','s','s','n','n','s','n','s','s','n','s','n','s','n','s','s','demócrata' ],
      ['s','s','n','s','s','s','n','n','n','s','n','s','s','s','n','s','republicano'],
      ['n','s','s','s','s','s','n','n','n','s','s','s','s','s','s','?','demócrata' ],
      ['s','s','s','n','s','s','n','n','?','s','n','n','n','s','s','?','demócrata' ],
      ['n','s','n','s','s','s','n','n','n','s','n','s','s','s','n','n','republicano'],
      ['s','?','s','n','n','n','s','s','s','n','?','n','n','n','s','?','demócrata' ],
      ['n','s','s','n','n','n','n','s','s','n','s','n','n','s','s','s','demócrata' ],
      ['n','n','s','n','n','n','s','s','s','n','n','n','n','n','s','?','demócrata' ],
      ['n','s','s','n','s','s','n','n','n','n','s','n','n','n','s','?','demócrata' ],
      ['s','n','s','n','n','n','s','s','s','n','s','n','n','n','s','?','demócrata' ],
      ['n','n','n','s','s','n','n','n','n','n','n','s','s','s','n','s','republicano'],
      ['n','s','n','s','s','s','n','n','n','s','n','?','s','s','n','n','republicano'],
      ['n','?','n','s','s','s','n','n','n','n','n','s','s','s','n','s','republicano'],
      ['n','n','s','n','n','s','s','s','s','n','s','n','n','s','s','s','demócrata' ],
      ['s','n','s','n','n','n','s','s','s','n','n','n','n','n','?','s','demócrata' ],
      ['n','s','n','s','s','s','n','n','n','n','n','s','s','?','n','s','republicano'],
      ['n','s','s','s','s','s','s','n','s','s','n','s','s','s','n','s','republicano'],
      ['n','s','n','s','s','s','n','n','n','n','n','s','s','s','n','s','republicano'],
      ['n','s','n','s','s','s','n','n','s','s','n','s','s','s','n','s','republicano'],
      ['n','s','s','n','n','n','s','s','n','n','s','n','n','n','s','?','demócrata' ],
      ['n','s','n','s','s','s','n','n','n','s','n','s','s','s','n','s','republicano'],
      ['n','n','s','n','n','s','s','s','s','s','n','s','n','s','s','?','demócrata' ],
      ['n','n','n','s','s','s','n','n','n','s','n','s','n','s','n','s','republicano'],
      ['n','n','s','n','n','n','s','s','s','n','n','n','n','n','s','s','demócrata' ],
      ['s','n','s','n','n','s','s','s','n','n','n','s','s','n','n','s','demócrata' ],
      ['s','s','s','n','n','n','s','s','?','s','n','n','n','n','s','?','demócrata' ],
      ['n','n','n','s','s','s','s','n','n','s','n','n','n','s','s','s','republicano'],
      ['n','n','n','s','n','s','s','?','s','n','n','s','s','s','n','s','republicano'],
      ['s','n','s','n','n','n','s','s','s','s','s','n','n','s','s','s','demócrata' ],
      ['n','n','n','n','s','s','s','n','n','n','n','?','n','s','s','s','republicano'],
      ['n','s','s','n','n','n','s','s','?','s','n','n','s','n','s','s','demócrata' ],
      ['s','n','s','n','n','n','n','s','s','s','n','n','n','n','s','s','demócrata' ],
      ['s','n','s','n','n','n','s','s','s','s','s','n','n','n','s','s','demócrata' ],
      ['n','n','s','n','s','n','s','s','s','n','n','n','n','s','?','s','demócrata' ],
      ['n','s','n','s','s','s','?','n','n','n','n','?','s','s','n','n','republicano'],
      ['?','?','?','?','?','?','?','?','?','?','?','?','?','?','?','?','republicano'],
      ['s','n','s','n','n','n','s','s','?','n','s','n','n','n','s','s','demócrata' ],
      ['n','s','n','s','s','s','n','n','n','n','n','s','s','s','n','n','republicano'],
      ['n','s','n','s','s','s','n','n','n','n','n','s','s','s','n','n','republicano'],
      ['s','s','s','n','n','s','s','s','s','n','n','n','n','n','s','s','demócrata' ],
      ['n','s','n','s','s','s','n','n','n','n','n','s','s','s','n','s','republicano'],
      ['s','n','s','n','n','n','s','s','s','s','n','n','n','n','n','s','demócrata' ],
      ['s','n','s','n','n','n','s','s','s','s','n','n','n','s','s','s','demócrata' ],
      ['n','n','n','s','s','n','n','n','n','n','n','s','n','s','n','n','republicano'],
      ['n','n','n','s','s','n','n','n','n','n','n','s','n','s','?','s','republicano'],
      ['n','n','s','n','n','n','s','s','s','n','s','n','n','n','s','s','demócrata' ],
      ['s','n','s','n','n','n','s','s','s','n','n','n','n','n','n','s','demócrata' ],
      ['s','n','s','n','n','n','s','s','s','s','n','n','n','n','n','s','demócrata' ],
      ['s','n','s','n','n','?','s','s','s','n','?','?','n','?','?','?','demócrata' ],
      ['s','n','s','n','n','n','s','s','s','s','n','n','?','n','s','s','demócrata' ],
      ['s','n','s','n','n','n','s','s','s','n','n','n','n','n','s','?','demócrata' ],
      ['s','n','s','n','n','n','s','s','s','n','n','n','n','n','s','?','demócrata' ],
      ['s','n','s','n','n','n','s','s','s','s','n','n','n','n','n','s','demócrata' ],
      ['n','n','n','s','s','s','n','n','n','s','n','s','n','s','n','s','republicano'],
      ['s','n','n','n','n','n','s','s','s','s','n','n','n','s','n','s','republicano'],
      ['s','n','s','n','n','n','s','s','s','n','n','n','n','n','s','?','demócrata' ],
      ['s','n','s','n','n','n','s','s','s','n','n','n','n','n','n','s','demócrata' ],
      ['s','s','s','n','n','n','s','s','s','n','n','n','n','n','s','s','demócrata' ],
      ['n','s','s','n','n','s','s','s','s','n','?','n','n','n','n','s','demócrata' ],
      ['s','n','s','n','n','n','s','s','s','s','n','n','n','n','s','?','demócrata' ],
      ['n','n','n','s','s','n','s','s','n','s','n','s','s','s','?','s','republicano'],
      ['s','n','n','s','s','n','s','n','n','s','n','n','n','s','s','s','republicano'],
      ['n','n','s','n','s','s','n','n','n','n','?','n','s','s','n','n','demócrata' ],
      ['n','n','n','s','s','s','n','n','n','n','n','s','s','s','s','n','republicano'],
      ['n','n','s','s','s','s','s','s','n','s','n','n','n','s','n','s','republicano'],
      ['n','n','n','s','s','s','n','n','n','n','n','s','s','s','n','s','republicano']]


# Clasificación de los ejemplos del conjunto de entrenamiemto:
# ------------------------------------------------------------




# Ejemplos del conjunto de validación:
# ------------------------------------


valid=[['n','n','n','s','s','s','n','n','n','s','n','s','s','s','n','n','republicano'],
       ['n','n','s','n','n','n','s','s','s','s','n','n','n','s','n','s','demócrata'  ],
       ['s','n','s','s','s','s','s','s','n','n','n','n','n','s','n','?','republicano'],
       ['s','n','n','s','s','s','n','n','n','s','n','?','s','s','n','n','republicano'],
       ['n','n','n','s','s','s','n','n','n','n','n','s','s','s','n','s','republicano'],
       ['n','n','s','n','n','s','s','s','s','s','s','n','n','n','?','s','demócrata'  ],
       ['n','n','s','n','n','s','s','s','s','s','s','n','n','n','s','s','demócrata'  ],
       ['n','n','s','n','n','s','?','s','?','s','s','s','n','s','s','?','demócrata'  ],
       ['s','s','s','?','n','s','s','s','s','n','s','n','s','n','?','s','demócrata'  ],
       ['s','s','s','n','s','s','n','s','n','s','s','n','s','s','s','s','demócrata'  ],
       ['s','s','s','n','s','s','n','s','n','s','s','n','s','s','n','?','demócrata'  ],
       ['s','n','s','n','?','s','?','s','s','s','n','n','s','s','n','s','demócrata'  ],
       ['s','n','s','n','n','s','s','s','s','s','n','?','n','s','n','s','demócrata'  ],
       ['s','n','s','n','n','s','s','s','n','s','s','n','s','s','s','s','demócrata'  ],
       ['s','s','s','n','n','s','s','s','s','s','s','n','s','s','s','s','demócrata'  ],
       ['n','s','s','n','n','s','s','s','n','s','s','n','s','s','n','?','demócrata'  ],
       ['n','s','n','s','s','s','?','?','n','s','n','s','?','?','?','?','republicano'],
       ['n','n','s','s','s','s','n','n','n','s','n','s','s','s','s','s','republicano'],
       ['s','s','s','n','n','s','s','s','s','s','n','n','?','n','s','?','demócrata'  ],
       ['n','s','n','n','n','n','s','s','s','s','s','n','n','n','s','s','demócrata'  ],
       ['n','s','s','n','n','s','s','s','s','s','n','n','s','s','s','s','demócrata'  ],
       ['n','n','n','s','s','n','s','s','s','s','n','s','s','s','n','s','republicano'],
       ['n','n','?','n','n','s','s','s','s','n','n','n','n','n','s','s','demócrata'  ],
       ['n','n','n','s','s','s','s','n','n','s','n','s','s','s','n','s','republicano'],
       ['n','n','n','s','s','s','n','n','n','n','n','s','s','s','n','n','republicano'],
       ['n','s','n','s','s','s','n','n','n','s','n','s','s','s','n','?','republicano'],
       ['n','n','n','s','s','s','n','n','n','s','n','s','s','s','n','n','republicano'],
       ['n','n','n','s','s','s','n','n','n','n','n','s','s','s','n','n','republicano'],
       ['s','n','s','n','n','s','s','s','s','n','n','n','n','s','n','?','demócrata'  ],
       ['n','n','n','s','s','s','n','n','n','s','n','s','s','s','n','n','republicano'],
       ['s','n','n','n','n','s','s','s','s','s','n','n','n','s','s','s','demócrata'  ],
       ['n','n','n','s','s','s','n','n','n','s','n','s','s','s','s','n','republicano'],
       ['n','n','s','n','n','s','s','s','s','s','n','n','s','n','n','s','demócrata'  ],
       ['s','s','s','n','n','n','s','s','s','s','n','n','n','n','s','s','demócrata'  ],
       ['n','s','s','s','s','s','n','n','n','s','n','s','s','s','n','s','republicano'],
       ['n','s','n','s','s','s','s','s','n','n','s','s','s','s','s','s','republicano'],
       ['n','s','s','s','s','s','s','?','n','n','n','n','?','?','s','?','republicano'],
       ['n','n','n','n','n','s','n','s','s','n','s','s','s','s','s','n','demócrata'  ],
       ['s','n','n','n','n','n','s','s','s','s','n','n','n','n','s','s','demócrata'  ],
       ['n','n','s','n','n','n','s','s','s','n','n','n','n','n','s','?','demócrata'  ],
       ['s','n','s','n','n','n','s','s','s','n','n','n','n','n','s','?','demócrata'  ],
       ['n','s','s','n','n','s','n','s','s','s','n','n','s','s','n','s','demócrata'  ],
       ['s','s','s','n','n','n','s','s','s','s','n','n','s','n','n','s','demócrata'  ],
       ['s','s','s','n','?','s','n','?','n','n','s','n','s','s','n','?','demócrata'  ],
       ['s','s','s','n','s','s','n','s','?','s','n','n','s','s','n','?','demócrata'  ],
       ['n','s','n','s','s','s','n','n','n','n','s','s','s','s','n','n','republicano'],
       ['n','s','n','n','s','s','n','n','?','n','n','s','s','s','n','s','demócrata'  ],
       ['s','s','n','s','n','n','s','s','s','n','s','n','n','s','n','s','demócrata'  ],
       ['n','s','n','s','s','s','n','n','n','n','n','s','s','s','n','s','republicano'],
       ['s','s','s','n','n','n','s','s','s','n','s','n','n','n','n','s','demócrata'  ],
       ['s','?','s','n','n','s','s','s','s','s','n','n','n','n','s','?','demócrata'  ],
       ['n','s','n','s','s','s','n','n','n','s','n','s','s','s','n','n','republicano'],
       ['s','?','s','n','n','n','s','s','s','n','n','n','n','n','s','?','demócrata'  ],
       ['s','n','s','n','n','n','s','s','s','n','s','n','n','n','s','?','demócrata'  ],
       ['n','n','s','n','n','n','s','s','s','n','n','n','n','n','s','s','demócrata'  ],
       ['n','s','s','n','n','s','s','s','?','n','s','s','n','n','s','s','demócrata'  ],
       ['n','n','n','s','s','s','n','n','n','s','s','s','s','s','n','?','republicano'],
       ['n','n','s','n','n','s','s','s','n','n','s','n','n','s','?','s','demócrata'  ],
       ['s','n','s','n','n','n','s','s','s','n','n','n','n','n','s','s','demócrata'  ],
       ['s','n','s','n','n','n','s','s','s','s','n','n','n','s','s','s','demócrata'  ],
       ['s','n','n','s','s','s','n','n','n','n','s','s','s','s','n','n','republicano'],
       ['n','n','n','s','s','s','n','n','n','s','s','s','n','s','n','s','republicano'],
       ['n','?','s','?','n','s','s','s','s','s','s','n','?','?','s','s','demócrata'  ],
       ['n','s','s','n','s','?','s','n','n','s','s','n','s','n','s','s','demócrata'  ],
       ['n','n','n','s','s','n','s','n','s','s','n','n','n','s','n','s','republicano'],
       ['n','n','s','n','n','n','s','s','s','s','s','n','n','n','s','s','demócrata'  ],
       ['n','n','n','s','s','s','s','n','n','s','n','s','n','s','s','s','republicano'],
       ['n','n','n','s','s','s','n','n','n','s','n','s','s','s','n','s','republicano'],
       ['s','n','n','s','s','s','n','n','n','s','n','s','s','s','n','n','republicano']]


# Ejemplos del conjunto de prueba (o test):
# =========================================


test=[['s','n','s','n','n','n','s','s','s','s','n','s','n','n','s','?','demócrata'  ],
      ['n','s','s','s','s','s','s','s','s','n','n','s','s','s','n','s','republicano'],
      ['n','s','n','n','n','s','s','n','s','n','s','n','n','n','s','s','demócrata'  ],
      ['n','n','s','s','s','s','s','s','s','s','n','s','s','s','s','s','republicano'],
      ['n','s','n','s','n','s','s','s','s','n','s','n','s','n','s','?','demócrata'  ],
      ['n','n','s','s','s','s','s','n','n','s','s','s','s','s','n','s','republicano'],
      ['n','s','s','n','n','s','s','s','s','s','n','?','n','n','s','s','demócrata'  ],
      ['s','n','s','s','n','n','n','s','s','s','n','n','n','s','s','s','republicano'],
      ['n','n','n','s','s','s','n','n','n','n','n','s','s','s','n','n','republicano'],
      ['n','n','n','s','s','s','n','n','n','n','n','s','s','s','n','n','republicano'],
      ['s','s','s','n','n','s','s','s','s','s','s','s','s','s','n','?','demócrata'  ],
      ['n','n','n','s','s','s','n','n','n','s','?','s','s','s','n','s','republicano'],
      ['s','n','s','n','n','s','s','s','s','s','n','n','s','n','n','s','demócrata'  ],
      ['s','n','s','n','s','s','s','n','s','s','n','n','s','s','n','?','demócrata'  ],
      ['s','s','s','n','n','s','s','s','s','s','s','s','s','n','n','s','demócrata'  ],
      ['s','s','n','s','s','s','n','n','n','s','s','n','s','n','n','n','republicano'],
      ['s','s','n','s','s','s','n','n','n','n','s','n','s','s','n','s','republicano'],
      ['n','s','n','n','s','s','n','n','n','s','s','n','s','s','n','n','demócrata'  ],
      ['s','n','s','n','n','n','s','s','n','s','s','n','n','n','n','?','demócrata'  ],
      ['s','s','s','n','s','s','s','s','n','s','s','n','n','n','s','?','demócrata'  ],
      ['n','s','s','n','n','s','s','s','n','s','n','n','n','n','s','s','demócrata'  ],
      ['n','s','n','s','s','s','n','n','n','n','n','n','s','s','n','s','republicano'],
      ['s','s','s','n','?','s','s','s','n','s','?','?','n','n','s','s','demócrata'  ],
      ['s','s','s','n','?','n','s','s','s','s','n','n','n','n','s','?','demócrata'  ],
      ['n','s','s','s','s','s','n','n','n','n','s','s','?','s','n','n','demócrata'  ],
      ['n','s','s','?','s','s','n','s','n','s','?','n','s','s','?','s','demócrata'  ],
      ['n','s','n','s','s','s','n','n','n','n','n','s','s','s','n','s','republicano'],
      ['n','s','n','s','s','s','n','n','n','n','s','s','n','s','n','n','demócrata'  ],
      ['s','?','s','n','n','n','s','s','s','n','s','n','n','n','s','s','demócrata'  ],
      ['n','s','n','s','s','s','?','?','n','n','?','?','s','?','?','?','republicano'],
      ['n','n','n','s','s','s','n','n','n','n','n','s','s','s','n','s','republicano'],
      ['n','n','n','s','s','s','n','n','n','n','n','s','s','s','n','s','republicano'],
      ['s','s','s','n','n','s','?','s','s','n','s','n','s','n','s','s','demócrata'  ],
      ['s','s','s','n','s','s','s','s','s','s','s','n','s','s','n','?','demócrata'  ],
      ['s','s','n','s','s','s','n','n','n','n','s','n','s','s','n','?','demócrata'  ],
      ['s','s','s','n','s','s','n','s','s','s','s','n','n','n','n','s','demócrata'  ],
      ['s','s','s','s','s','s','n','n','n','n','s','s','s','s','n','s','demócrata'  ],
      ['s','s','n','n','s','s','n','n','n','n','s','s','s','s','s','n','demócrata'  ],
      ['n','?','s','n','s','s','n','s','n','n','s','n','n','n','n','?','demócrata'  ],
      ['s','s','s','n','s','s','n','s','s','n','s','n','n','s','n','?','demócrata'  ],
      ['n','s','s','s','s','s','n','n','n','n','n','s','s','s','n','?','demócrata'  ],
      ['s','n','s','n','n','n','s','s','s','?','s','n','n','n','s','?','demócrata'  ],
      ['?','?','n','n','?','s','?','n','n','n','s','s','n','s','n','?','demócrata'  ],
      ['s','s','n','n','n','n','n','s','s','n','s','n','n','n','s','n','demócrata'  ],
      ['s','s','n','s','s','s','n','n','n','n','s','s','s','s','n','s','republicano'],
      ['?','?','?','?','n','s','n','s','s','n','n','s','s','n','n','?','republicano'],
      ['s','s','?','?','?','s','n','n','n','n','s','n','s','n','n','s','demócrata'  ],
      ['s','s','s','?','n','n','n','s','n','n','s','?','n','n','s','s','demócrata'  ],
      ['s','s','s','n','s','s','n','s','n','n','s','n','s','n','s','s','demócrata'  ],
      ['s','s','n','n','s','?','n','n','n','n','s','n','s','s','n','s','demócrata'  ],
      ['n','s','s','n','s','s','n','s','n','n','n','n','n','n','n','s','demócrata'  ],
      ['n','s','n','s','?','s','n','n','n','s','n','s','s','s','n','n','republicano'],
      ['n','s','n','s','s','s','n','?','n','n','?','?','?','s','n','?','republicano'],
      ['n','s','n','s','s','s','n','n','n','s','s','s','s','s','n','n','republicano'],
      ['?','n','s','s','n','s','s','s','s','s','n','s','n','s','n','s','republicano'],
      ['n','s','n','s','s','s','n','n','n','s','n','s','?','s','n','n','republicano'],
      ['s','s','n','s','s','s','n','n','n','s','n','s','s','s','n','s','republicano'],
      ['n','n','n','s','s','s','n','n','n','n','n','s','s','s','n','s','republicano'],
      ['s','n','s','n','s','s','n','n','s','s','n','n','s','s','n','s','demócrata'  ],
      ['n','n','n','s','s','s','n','n','n','n','s','s','s','s','n','n','demócrata'  ],
      ['s','n','s','n','n','s','s','s','s','n','n','s','?','s','s','s','demócrata'  ],
      ['n','n','n','s','s','s','n','n','n','n','n','s','s','s','n','n','republicano'],
      ['n','n','n','s','s','s','n','n','n','n','s','s','s','s','n','s','republicano'],
      ['s','n','s','n','n','s','s','s','s','s','s','n','n','n','n','s','demócrata'  ],
      ['n','n','n','s','s','s','n','n','n','s','n','s','s','s','n','s','republicano'],
      ['s','s','s','s','s','s','s','s','n','s','?','?','?','s','n','s','republicano'],
      ['s','s','s','n','n','n','s','s','s','n','n','n','n','n','n','s','demócrata'  ],
      ['n','s','s','n','n','s','s','s','?','s','n','n','n','n','n','s','demócrata'  ],
      ['s','s','n','s','s','s','n','n','n','s','n','n','s','s','n','s','republicano'],
      ['s','s','s','n','n','n','s','s','s','s','s','n','s','n','n','s','demócrata'  ],
      ['s','s','s','n','n','n','s','s','n','s','n','n','n','n','n','s','demócrata'  ],
      ['s','s','s','n','n','n','s','s','s','n','n','n','n','n','n','s','demócrata'  ],
      ['s','s','s','s','s','s','s','s','n','s','n','n','s','s','n','s','republicano'],
      ['n','s','s','n','s','s','s','s','n','n','s','n','s','n','s','s','demócrata'  ],
      ['n','n','s','n','n','s','s','s','s','n','s','n','n','n','s','s','demócrata'  ],
      ['n','s','s','n','n','s','s','s','s','n','s','n','n','s','s','s','demócrata'  ],
      ['n','s','s','n','n','?','s','s','s','s','s','n','?','s','s','s','demócrata'  ],
      ['n','n','s','n','n','n','s','s','n','s','s','n','n','n','s','?','demócrata'  ],
      ['s','n','s','n','n','n','s','s','s','s','n','n','n','n','s','s','demócrata'  ],
      ['n','n','n','s','s','s','s','s','n','s','n','s','s','s','n','s','republicano'],
      ['?','?','?','n','n','n','s','s','s','s','n','n','s','n','s','s','demócrata'  ],
      ['s','n','s','n','?','n','s','s','s','s','n','s','n','?','s','s','demócrata'  ],
      ['n','n','s','s','s','s','n','n','s','s','n','s','s','s','n','s','republicano'],
      ['n','n','s','n','n','n','s','s','s','s','n','n','n','n','n','s','demócrata'  ],
      ['n','?','n','s','s','s','n','n','n','n','s','s','s','s','n','s','republicano'],
      ['n','n','n','s','s','s','?','?','?','?','n','s','s','s','n','s','republicano'],
      ['n','s','n','s','s','s','n','n','n','s','n','s','s','s','?','n','republicano']]

                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
