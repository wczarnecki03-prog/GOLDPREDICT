# GoldPredict: Gold Price Forecasting App 

Gold Price to aplikacja webowa stworzona w środowisku Python, która umożliwia prognozowanie przyszłych cen złota przy użyciu zaawansowanego modelowania szeregów czasowych SARIMA. System integruje się z danymi giełdowymi w czasie rzeczywistym, oferując precyzyjne analizy dla inwestorów i pasjonatów rynków finansowych.

## Kluczowe Funkcjonalności

Wybór Instrumentu: Obsługa kontraktów terminowych futures (GC=F) oraz funduszy ETF (GLD).
Elastyczność Jednostek: Możliwość przeliczania cen między USD za uncję (oz) a USD za gram (g).
Interaktywny Kalendarz: Samodzielny wybór horyzontu czasowego prognozy oraz zakresu danych historycznych.
Tryby Optymalizacji:
    Tryb Szybki: Ograniczenie przeszukiwania parametrów modelu dla błyskawicznych wyników.
    Agregacja Tygodniowa: Resampling danych do interwałów piątkowych, co przyspiesza     uczenie modelu o ok. 45–60%.
Wizualizacja Danych: Czytelne wykresy prezentujące historię, prognozę oraz pasmo niepewności (przedział ufności).

## Architektura i Technologie

| Aplikacja opiera się na nowoczesnym stosie technologicznym Python: |
| :--- |
 | Frontend: Streamlit – responsywny interfejs użytkownika z ciemnym motywem.|
 | Dane:   yfinance – pobieranie danych historycznych z Yahoo Finance.|
 | Silnik Statystyczny: statsmodels – implementacja modelu SARIMA.|
 | Matematyka i Logika: NumPy, Pandas oraz matplotlib do generowania wykresów.|

## Przetwarzanie Danych

Aplikacja stabilizuje wariancję szeregu czasowego poprzez operację na logarytmach:
$$y = \ln(\text{price})$$.
Następnie automatycznie eliminuje brakujące wartości i ujednolica formaty dat, aby zapewnić ciągłość szeregu.

## Prezentacja Systemu

Uwaga techniczna: Wynik prezentowany jest jako interaktywny wykres oraz wyróżniona metryka z końcową prognozowaną wartością. 
Wszystkie szczegóły modelu (AIC, parametry) są dostępne w sekcji technicznej.

## Instalacja i Uruchomienie

| Wymaganie | Specyfikacja Minimalna | Specyfikacja Zalecana |
|:--- | :--- | :--- |
| Procesor | 2 rdzenie (x64) | 4+ rdzenie, 2.5 GHz+ |
| RAM | 4 GB | 8 GB |
| Dysk | 1 GB wolnego miejsca | 2 GB wolnego miejsca |
| System | Windows 10/11 / Linux / macOS | Windows 11 / macOS |
  
## Instrukcja

1. Sklonuj repozytorium lub pobierz pliki źródłowe.
2. Zainstaluj wymagane biblioteki:

    | Bash |
    | :--- |
    |pip install streamlit yfinance statsmodels pandas matplotlib|

3. Uruchom aplikację za pomocą terminala:
| Bash |
| :--- |
| treamlit run app.py |

## Wyniki Testów (v1.0)

| Projekt przeszedł pełną walidację scenariuszy testowych (20.01.2026): |
| :--- |
|  GP_01 (Podstawowa prognoza): | Zaliczony. |
|  GP_02 (Zmiana źródła danych): Zaliczony. |
| GP_05 (Wydajność): Potwierdzono znaczące skrócenie czasu pracy w trybach optymalizacji. |
  
## Autorzy

| Sławomir Kobyłko | (Logika certyfikatów i SSL) |
| Miłosz Furman | (Stylizacja UI i CSS) |
| Wojciech Czarnecki (Struktura danych i wybór dat) |

Disclaimer
Aplikacja ma charakter wyłącznie edukacyjny. Prognozy generowane przez model SARIMA bazują jedynie na danych historycznych i nie uwzględniają czynników makroekonomicznych – nie stanowią one porady inwestycyjnej.
