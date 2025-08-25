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
  late Future<(List<Movie>, List<Movie>)> _futureData;

  @override
  void initState() {
    super.initState();
    _futureData = _load();
  }

  Future<(List<Movie>, List<Movie>)> _load() async {
    final recs = await ApiClient.fetchRecommendations(widget.userId);
    final similar = await ApiClient.fetchSimilar('Toy Story'); // placeholder X
    return (recs, similar);
  }

  void _openDetails(Movie m) {
    MovieDetailSheet.show(context, m, widget.userId);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF141414),
      body: SafeArea(
        child: FutureBuilder<(List<Movie>, List<Movie>)>(
          future: _futureData,
          builder: (context, snap) {
            if (snap.connectionState == ConnectionState.waiting) {
              return const Center(child: CircularProgressIndicator());
            }
            if (snap.hasError) {
              return Center(
                child: Text('Errore: ${snap.error}', style: const TextStyle(color: Colors.redAccent)),
              );
            }
            final (recs, similar) = snap.data ?? (<Movie>[], <Movie>[]);
            return ListView(
              children: [
                const SizedBox(height: 20),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 24),
                  child: Text(
                    'Benvenuto, ${widget.userId}',
                    style: const TextStyle(color: Colors.white, fontSize: 26, fontWeight: FontWeight.w800),
                  ),
                ),
                const SizedBox(height: 10),
                SectionRow(
                  title: 'Consigliati per te',
                  movies: recs,
                  onTapMovie: _openDetails,
                ),
                SectionRow(
                  title: 'Perché ti piace Toy Story',
                  movies: similar,
                  onTapMovie: _openDetails,
                ),
                const SizedBox(height: 30),
              ],
            );
          },
        ),
      ),
    );
  }
}
