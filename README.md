# Prepoznavanje emocija na osnovu slike lica
Projekat se bavi problemom prepoznavanja emocija na osnovu slika lica. Metode korišćene za rešavanje ovog problema su metoda potpornih vektora (eng. *Support Vector Machine*, **SVM**) i konvolucijske neuronske mreže (eng. *Convolutional Neural Network*, **CNN**). Za treniranje su korišćeni CK+ i FER2013 skupovi podataka. Rezultati dobijeni nad ovim skupovima su prikazani u tabeli.

|   |CK+   |FER2013|
|---|------|-------|
|CNN|94.47%|58.31% |
|SVM|87.7% |49.8%  |

## Potrebne biblioteke
Za uspešan rad potrebno imati Python 3.8.
Pored toga, neophodne su i sledeće biblioteke:
- Keras
- Tensorflow (2.4.1)
	> Prilikom instaliranja Tensorflow-a će se, između ostalih, instalirati i Pillow biblioteka. Postoji mogućnost da trenutna verzija Pillow biblioteke (8.3.0) neće biti kompatibilna sa nekim funkcionalnostima Keras-a, pa je potrebno smanjiti verziju na 8.2.0.
- openCV
- numpy
- pandas
- matplotlib
- dlib
- sklearn
- scikit-image

## Pokretanje

Za treniranje je potrebno skinuti [ck+](https://www.dropbox.com/s/b1nk6ob0vhyjvyu/ck%2B.zip?dl=0) i [fer2013](a) skupove i smestiti ih u glavni folder projekta

Takođe, potrebno je skinuti 
- [haarcascade_frontalface_default.xml](https://www.dropbox.com/s/4spo9m48qjvp8lh/haarcascade_frontalface_default.xml?dl=0)
- [shape_predictor_68_face_landmarks.dat](a)  

i smestiti u *svm* folder projekta.

### CNN

#### Treniranje
 
Da bi se pokrenulo treniranje mreže potrebno je pokrenuti fajl `train.py` sledećom komandom.

`python train.py`

Ako korisnik želi da promeni dataset nad kojim se izvršava treniranje potrebno je da promeni liniju 92 unutar `trani.py` fajla na sledeći način, u zavisnosti od dataseta: 

- CK+

> _data = load_data('ck')

- FER2013

> _data = load_data('fer')

Model će po završetku uspešnog treniranja biti sačuvan na sledećoj putanji:

> cnn/model_with_[dataset].h5

Gde dataset može biti 'ck' ili 'fer' 
#### Testiranje 
Evaluacija se pokreće uz pomoć komande

 `python predict.py`

Ova skripta će pokazati tačnost istreniranog modela u zavisnosti od dataseta. Ako korisnik želi da promeni dataset nad kojim se vrši provera mora da izmeni liniju u 43 fajlu `predict.py` na sledeći način: 

- CK+

> evaluate("ck")

- FER2013

> evaluate("fer")
> 
### SVM
Parametri (skup koji se koristi, broj slika po emociji...) se nalaze u fajlu *svm_params.py*. Potrebno je uneti odgovarajuće parametre pre izdvajanje karakteristika lica i treniranja.
#### Izdvajanje karakteristika lica
Izdvajanje karakteristika lica se vrši pokretanjem komande `python feature_extraction.py`
Ovim se kreira folder koji sadrži neophodne fajlove za treniranje SVM-a.
#### Treniranje
Nakon izdvajanja karakteristika lica, potrebno je pokrenuti treniranje. Pokretanje se vrši komandom `python train.py`. Mogući argumenti komandne linije su:

- -t (\-\-train) - Može imati 'yes' ili 'no' vrednosti. Trenira SVM na osnovu karakteristika lica. Podrazumevano uključeno
- -e (\-\-evaluate) - Može imati 'yes' ili 'no' vrednosti. Učitava model i vrši evaluaciju. Podrazumevano isključeno
### Real-time prepoznavanje emocija
Za *real-time* prepoznavanje emocija je poterbno da su zadovoljeni gornji uslovi. Pokretanje se vrši komandom `python real-time-recognition.py -m=svm`
> Parametar -m (\-\-method) može imati vrednosti 'svm' i 'cnn'. Podešavanje skupa podataka se vrši u fajlu svm_params.py, kao i kod treniranja. Podrazumevani metod je CNN.

