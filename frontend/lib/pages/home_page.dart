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

class HomeData {
  final String likedTitle;
  final List<Movie> recs;
  const HomeData({required this.likedTitle, required this.recs});
}

class _HomePageState extends State<HomePage> {
  late Future<HomeData> _futureData;

  @override
  void initState() {
    super.initState();
    _futureData = _load();
  }

  Future<HomeData> _load() async {
    // 1) preferenze utente per ricavare liked_movie
    final prefs = await ApiClient.fetchUserPreferences(widget.userId);
    final likedTitle = (prefs['liked_movie'] as String?)?.trim();
    final seedTitle = (likedTitle == null || likedTitle.isEmpty)
        ? 'Toy Story'
        : likedTitle;

    // 2) una sola chiamata all'endpoint ibrido (10 film)
    final raw = await ApiClient.fetchRecommendations(
      widget.userId,
      useSampled: true, // false => top deterministici
    );

    // 3) CINTURA: filtra qualsiasi voce nulla o senza titolo
    final List<Movie> recs = raw
        .whereType<Movie>() // scarta eventuali null (nel caso arrivino…)
        .where((m) => m.title.trim().isNotEmpty)
        .toList(growable: false);

    // debug rapido: vedi cosa stai mostrando
    // debugPrint('HYBRID MOVIES: ${recs.map((m) => m.title).toList()}');

    return HomeData(likedTitle: seedTitle, recs: recs);
  }

  Future<void> _refresh() async {
    setState(() => _futureData = _load());
    await _futureData;
  }

  void _openDetails(Movie m) {
    // protezione extra: se mai arrivasse qualcosa di strano
    if (m.title.trim().isEmpty) return;
    MovieDetailSheet.show(context, m, widget.userId);
  }

  @override
  Widget build(BuildContext context) {
    const bg = Color(0xFF141414);
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
        child: FutureBuilder<HomeData>(
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
                  textAlign: TextAlign.center,
                ),
              );
            }
            if (!snap.hasData) {
              return const SizedBox.shrink();
            }

            final home = snap.data!;
            final String likedTitle = home.likedTitle;
            final List<Movie> recs = home.recs;

            final Widget section = recs.isEmpty
                ? const Padding(
                    padding: EdgeInsets.all(24),
                    child: Text(
                      'Nessun consiglio disponibile al momento.',
                      style: TextStyle(color: Colors.white70),
                    ),
                  )
                : SectionRow(
                    title: 'Consigliati per te',
                    titleStyle: sectionTitleStyle,
                    movies: recs, // List<Movie> pulita (no null)
                    onTapMovie: _openDetails,
                    showArrows: true,
                    showExploreBadge: false,
                  );

            return RefreshIndicator(
              onRefresh: _refresh,
              child: ListView(
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
                          Text(
                            'Benvenuto, ${widget.userId}',
                            style: titleStyle,
                          ),
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

                  section,

                  const SizedBox(height: 28),
                ],
              ),
            );
          },
        ),
      ),
    );
  }
}
