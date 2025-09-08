import 'package:flutter/material.dart';
import '../models/movie.dart';
import '../services/api_client.dart';
import '../widgets/section_row.dart';
import 'movie_detail_sheet.dart';

class HomePage extends StatefulWidget {
  final String userId;
  const HomePage({super.key, required this.userId});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  // Ora carichiamo: likedTitle, recs, similar, banditExploit, banditExplore
  late Future<(String, List<Movie>, List<Movie>, List<Movie>, List<Movie>)> _futureData;

  @override
  void initState() {
    super.initState();
    _futureData = _load();
  }

  Future<(String, List<Movie>, List<Movie>, List<Movie>, List<Movie>)> _load() async {
    // 1) preferenze utente per ricavare liked_movie
    final prefs = await ApiClient.fetchUserPreferences(widget.userId);
    final likedTitle = (prefs['liked_movie'] as String?)?.trim();
    final seedTitle = (likedTitle == null || likedTitle.isEmpty) ? 'Toy Story' : likedTitle;

    // 2) chiamate in parallelo
    final recsF = ApiClient.fetchRecommendations(widget.userId);
    final similarF = ApiClient.fetchSimilar(seedTitle);
    final banditF = ApiClient.fetchBandit(
      widget.userId,
      topK: 20,
      epsilon: 0.35,
      candidatePool: 160,
      exploreExtra: 400,
      seed: 42, // stabile in dev
    );

    final recs = await recsF;
    final similar = await similarF;
    final (banditExploit, banditExplore) = await banditF;

    return (seedTitle, recs, similar, banditExploit, banditExplore);
  }

  void _openDetails(Movie m) {
    MovieDetailSheet.show(context, m, widget.userId);
  }

  @override
  Widget build(BuildContext context) {
    const bg = Color(0xFF141414); // Netflix dark
    const titleStyle = TextStyle(
      color: Colors.white,
      fontSize: 28,
      fontWeight: FontWeight.w900,
      letterSpacing: 0.2,
    );
    const sectionTitleStyle = TextStyle(
      color: Colors.white,
      fontSize: 20,
      fontWeight: FontWeight.w800,
    );
    const subtitleStyle = TextStyle(
      color: Colors.white70,
      fontSize: 13,
      fontWeight: FontWeight.w500,
      letterSpacing: 0.2,
    );

    return Scaffold(
      backgroundColor: bg,
      body: SafeArea(
        child: FutureBuilder<(String, List<Movie>, List<Movie>, List<Movie>, List<Movie>)>(
          future: _futureData,
          builder: (context, snap) {
            if (snap.connectionState == ConnectionState.waiting) {
              return const Center(
                child: CircularProgressIndicator(color: Colors.redAccent),
              );
            }
            if (snap.hasError) {
              return Center(
                child: Text(
                  'Errore: ${snap.error}',
                  style: const TextStyle(color: Colors.redAccent),
                ),
              );
            }
            final (likedTitle, recs, similar, banditExploit, banditExplore) =
                snap.data ?? ('Toy Story', <Movie>[], <Movie>[], <Movie>[], <Movie>[]);

            return ListView(
              padding: EdgeInsets.zero,
              children: [
                // HEADER HERO
                Container(
                  height: 140,
                  decoration: const BoxDecoration(
                    gradient: LinearGradient(
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                      colors: [Color(0xFF1A1A1A), Color(0xFF0E0E0E)],
                    ),
                  ),
                  child: Padding(
                    padding: const EdgeInsets.fromLTRB(24, 24, 24, 16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('Benvenuto, ${widget.userId}', style: titleStyle),
                        const SizedBox(height: 8),
                        Text(
                          likedTitle.isNotEmpty
                              ? 'Consigli basati anche su: $likedTitle'
                              : 'Scopri titoli su misura e novità',
                          style: subtitleStyle,
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 8),

                // baseline
                SectionRow(
                  title: 'Consigliati per te',
                  titleStyle: sectionTitleStyle,
                  movies: recs,
                  onTapMovie: _openDetails,
                  showArrows: true,
                  showExploreBadge: true, // innocuo (nessuno è explore qui)
                ),

                // similar by dynamic title
                SectionRow(
                  title: 'Perché ti piace $likedTitle',
                  titleStyle: sectionTitleStyle,
                  movies: similar,
                  onTapMovie: _openDetails,
                  showArrows: true,
                ),

                // ====== SEZIONE BANDIT (esplorazione) ======
                const SizedBox(height: 8),
                SectionRow(
                  title: 'Esplora (novità)',
                  titleStyle: sectionTitleStyle,
                  accentColor: const Color(0xFFE50914), // rosso Netflix
                  movies: banditExplore,
                  onTapMovie: _openDetails,
                  showArrows: true,
                  showExploreBadge: true, // badge “NOVITÀ” sugli explore
                ),

                const SizedBox(height: 28),
              ],
            );
          },
        ),
      ),
    );
  }
}
