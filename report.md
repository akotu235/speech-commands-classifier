### Sztuczna Inteligencja i Systemy Ekspertowe
# Sieci Neuronowe
## *Zastosowanie sieci neuronowych w wybranym problemie*
## [**Andrzej Kotulski**](https://akotu235.github.io/)
[**WSZiB**](https://www.wszib.edu.pl/), 2024

---


## 1. Wprowadzenie
<!---
[Celem projektu jest zastosowanie sieci neuronowych do rozwiązania wybranego problemu. W niniejszym raporcie zostanie przedstawiony problem, wybrane podejście oraz wyniki analizy.]
-->

Celem projektu jest zastosowanie sieci neuronowych do rozwiązania problemu rozpoznawania mowy, polegającego na&nbsp;klasyfikacji nagrań dźwiękowych. Rozpoznawanie mowy stanowi kluczowy obszar badań w&nbsp;dziedzinie sztucznej inteligencji i&nbsp;znajduje szerokie zastosowanie m.in.&nbsp;w&nbsp;systemach asystentów głosowych oraz urządzeniach IoT [[1]](#bibliografia).

W niniejszym raporcie zostanie przedstawiona implementacja modelu sieci neuronowej z&nbsp;wykorzystaniem biblioteki TensorFlow [[3]](#bibliografia) oraz ekstrakcją cech dźwiękowych za&nbsp;pomocą narzędzia Librosa [[2]](#bibliografia).

## 2. Definicja problemu
<!---
[Opis problemu:
*	Rodzaj problemu: (np. klasyfikacja, predykcja, segmentacja)
*	Dlaczego jest to ważne?]
-->

**Opis problemu**:

Rozpoznawanie mowy polega na&nbsp;klasyfikacji nagrań dźwiękowych do&nbsp;odpowiednich kategorii (np.&nbsp;komend głosowych). Jest to&nbsp;klasyczny problem **klasyfikacji wieloklasowej**, który może być rozwiązany za&nbsp;pomocą sieci neuronowych, takich jak modele oparte na&nbsp;warstwach konwolucyjnych (CNN) lub rekurencyjnych (LSTM).

**Znaczenie problemu**:

Technologie rozpoznawania mowy, oparte na&nbsp;sieciach głębokich, przyczyniły się do&nbsp;rozwoju asystentów głosowych i&nbsp;systemów wspomagania. Dzięki temu możliwe jest tworzenie bardziej intuicyjnych interfejsów dla użytkowników oraz wspieranie osób z&nbsp;ograniczeniami ruchowymi.

## 3. Użyte narzędzia i biblioteki
<!---
[Lista narzędzi i bibliotek:
* Framework: (np. PyTorch, TensorFlow).
*	Inne biblioteki: (np. NumPy, pandas).]
-->

Do realizacji projektu wykorzystano następujące narzędzia i&nbsp;biblioteki:

**Framework**:

* **TensorFlow/Keras** – do budowy, trenowania i&nbsp;testowania modeli sieci neuronowych [[3]](#bibliografia).

**Biblioteki do przetwarzania danych**:

* **NumPy** – operacje na&nbsp;tablicach i&nbsp;macierzach [[5]](#bibliografia).
* **Pandas** – manipulacja i&nbsp;analiza danych strukturalnych [[6]](#bibliografia).
* **Librosa** – ekstrakcja cech dźwiękowych (m.in.&nbsp;MFCC – Mel-Frequency Cepstral Coefficients) [[2]](#bibliografia).

**Środowisko uruchomieniowe**:

* **Google Colab** – platforma do&nbsp;uruchamiania kodu w&nbsp;chmurze z&nbsp;możliwością wykorzystania GPU [[7]](#bibliografia).

**Biblioteki do wizualizacji wyników**:

* **Matplotlib** – generowanie wykresów do&nbsp;analizy wyników [[8]](#bibliografia).
* **Seaborn** – wizualizacja danych wejściowych i&nbsp;wyników modelu [[9]](#bibliografia).

## 4. Przygotowanie zbioru danych
<!---
[Szczegóły dotyczące przygotowania danych:
*	Źródło danych: (np. Kaggle, UCI Repository).
*	Podział na zbiór treningowy, walidacyjny i testowy.
*	Proces wstępnego przetwarzania danych (np. normalizacja, augmentacja).]
-->

## 5. Szczegóły modelu
<!---
[Opis zastosowanego modelu:
  *	Architektura modelu: (np. CNN, LSTM).
  *	Parametry modelu: (np. liczba warstw, funkcja aktywacji).]
-->

## 6. Wyniki i ocena
<!---
[Metryki oceny:
*	Dokładność, F1-score, Recall itp.
Wyniki uzyskane na zbiorze walidacyjnym i testowym.]
-->

## 7. Wnioski
<!---
[Podsumowanie projektu, porównanie wyników oraz propozycje dalszych prac.]
-->

## Bibliografia

1. **F. Chollet, Deep Learning with Python**, Manning Publications, 2018.

2. **Librosa: Biblioteka do analizy dźwięku w Python**. Dostępne online: <https://librosa.org>.

3. **TensorFlow/Keras: Dokumentacja API**. Dostępne online: <https://www.tensorflow.org>.

4. **Google Speech Commands Dataset, Kaggle**. Dostępne online: <https://www.kaggle.com/datasets>.

5. **NumPy: Dokumentacja biblioteki do operacji na tablicach**. Dostępne online: https://numpy.org.

6. **Pandas: Dokumentacja biblioteki do analizy danych**. Dostępne online: https://pandas.pydata.org.

7. **Google Colab: Platforma do uruchamiania kodu w chmurze**. Dostępne online: https://colab.research.google.com.

8. **Matplotlib: Biblioteka do wizualizacji danych**. Dostępne online: https://matplotlib.org.

9. **Seaborn: Biblioteka do wizualizacji statystycznej**. Dostępne online: https://seaborn.pydata.org.


## Załączniki

1. **Repozytorium kodu źródłowego** – Pełny kod projektu. Dostępne online: <https://github.com/akotu235/sieci-neuronowe>.

2. **Wersja online sprawozdania** – Bieżąca wersja dokumentu. Dostępne online: <https://github.com/akotu235/sieci-neuronowe/blob/master/report/report.md>.