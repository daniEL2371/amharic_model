vocab = list(set(open('data/news.txt', encoding='utf8').read().split(' ')))
words = {}
for v in vocab:
    words[v] = v

city = [
    ('ጎንደር', 'አማራ'),
    ('ደሴ', 'አማራ'),
    ('ወልድያ', 'አማራ'),
    ('ኮምቦልቻ', 'አማራ'),
    ('አዳማ', 'ኦሮምያ'),
    ('ሻሸመኔ', 'ኦሮምያ'),
    ('አሪሲ', 'ኦሮምያ'),
    ('አምቦ', 'ኦሮምያ'),
    ('ቢሾፍቱ', 'ኦሮምያ'),
    ('ጅማ', 'ኦሮምያ'),
    ('መቀሌ', 'ትግራይ'),
    ('አዋሳ', 'ደቡብ'),
    ('ዲላ', 'ደቡብ'),
    ('ሶዶ', 'ደቡብ'),
    ('ይርጋለም', 'ደቡብ'),
    ('ባሌ', 'ኦሮምያ'),
    ('ወለጋ', 'ኦሮምያ'),
    ('አዲግራት', 'ትግራይ'),
    ('አድዋ', 'ትግራይ'),
    ('ሽሬ', 'ትግራይ'),
    ('አላማጣ', 'ትግራይ'),
    ('አክሱም', 'ትግራይ'),
]
party =[
    ('ሕውሓት', 'ትግራይ'),
    ('ደሕዴን', 'ደቡብ'),
    ('ብአዴን', 'አማራ'),
    ('ኦሕዴድ', 'ኦሮሚያ'),
]
beer =[
    ('ዳሽን', 'ቢራ'),
    ('በዴሌ', 'ቢራ'),
    ('ሐረር', 'ቢራ'),
    ('ዋለያ', 'ቢራ'),
    ('ሔኒከን', 'ቢራ'),
    ('ሚሪንዳ', 'ለስላሳ'),
    ('ኮካኮላ', 'ለስላሳ'),
    ('ፔፕሲ', 'ለስላሳ'),
    ('ፋንታ', 'ለስላሳ'),
]

holiday =[
    ('ገና', 'ክርስቲያን'),
    ('ስቅለት', 'ክርስቲያን'),
    ('ጥምቀት', 'ክርስቲያን'),
    ('ፋሲካ', 'ክርስቲያን'),
    ('ረመዳን', 'እስላም'),
    ('አልፈጥር', 'እስላም'),
    ('አረፋ', 'እስላም'),
    ('ሙባረክ', 'እስላም'),
]

trans = [
    ("ጀልባ", 'ውሀ'),
    ("መርከብ", 'ውሀ'),
    ("መኪና", 'የብስ'),
    ("ጀት", 'አየር'),
    ("ሔሊኮፕተር", 'አየር'),
]
gender = [
    ("", "")
]
l = [trans, party, holiday, city, beer]
analogy = []
for ll in l:
    for pair in ll:
        a, b = pair
        if a in words and b in words:
            for pair2 in city:
                c, d = pair2
                if c in words and d in words and d != b:
                    analogy.append((a, b, c, d))

s = ""
for a in analogy:
    s += "{0} {1} {2} {3}\n".format(a[0], a[1], a[2], a[3])
open('new.txt', encoding='utf8', mode='w').write(s)
                
