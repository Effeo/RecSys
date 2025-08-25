import 'package:flutter/material.dart';
import '../models/movie.dart';
import '../services/api_client.dart';

class MovieDetailSheet extends StatefulWidget {
  final Movie movie;
  final String userId;

  const MovieDetailSheet({super.key, required this.movie, required this.userId});

  static Future<void> show(BuildContext context, Movie movie, String userId) {
    return showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: const Color(0xFF1A1A1A),
      builder: (_) => FractionallySizedBox(
        heightFactor: 0.92,
        child: MovieDetailSheet(movie: movie, userId: userId),
      ),
    );
  }

  @override
  State<MovieDetailSheet> createState() => _MovieDetailSheetState();
}

class _MovieDetailSheetState extends State<MovieDetailSheet> {
  int _rating = 0;
  bool _submitting = false;

  @override
  Widget build(BuildContext context) {
    final m = widget.movie;

    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  m.title,
                  style: const TextStyle(fontSize: 22, fontWeight: FontWeight.w800, color: Colors.white),
                ),
                IconButton(
                  icon: const Icon(Icons.close, color: Colors.white70),
                  onPressed: () => Navigator.pop(context),
                ),
              ],
            ),
            const SizedBox(height: 14),

            // Layout principale
            Expanded(
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Poster
                  ClipRRect(
                    borderRadius: BorderRadius.circular(8),
                    child: Image.asset(
                      'assets/poster_placeholder.jpg',
                      width: 260,
                      height: 260 * 9/16,
                      fit: BoxFit.cover,
                    ),
                  ),
                  const SizedBox(width: 24),
                  // Info
                  Expanded(
                    child: SingleChildScrollView(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          _infoRow('Uscita', m.releaseDate ?? 'n/d'),
                          _infoRow('Regista', m.director ?? 'n/d'),
                          _infoRow('Runtime', (m.runtime?.toString() ?? 'n/d')),
                          if (m.awards != null) _infoRow('Premi', '${m.awards}'),
                          if (m.score != null) _infoRow('Score', m.score!.toStringAsFixed(2)),
                          if (m.similarity != null) _infoRow('Similarità', m.similarity!.toStringAsFixed(2)),
                          if (m.imdbUrl != null)
                            Padding(
                              padding: const EdgeInsets.only(top: 8),
                              child: SelectableText(
                                m.imdbUrl!,
                                style: const TextStyle(color: Colors.blueAccent),
                              ),
                            ),
                          const SizedBox(height: 24),
                          const Text('Valuta questo titolo', style: TextStyle(color: Colors.white, fontSize: 16)),
                          const SizedBox(height: 8),
                          Row(
                            children: List.generate(5, (i) {
                              final idx = i + 1;
                              final filled = _rating >= idx;
                              return IconButton(
                                onPressed: () => setState(() => _rating = idx),
                                icon: Icon(
                                  filled ? Icons.star : Icons.star_border,
                                  color: filled ? Colors.amber : Colors.white70,
                                  size: 28,
                                ),
                              );
                            }),
                          ),
                          const SizedBox(height: 12),
                          FilledButton(
                            onPressed: (_rating == 0 || m.movieId == null || _submitting)
                                ? null
                                : () async {
                                    setState(() => _submitting = true);
                                    final ok = await ApiClient.submitRating(
                                      userId: widget.userId,
                                      movieId: m.movieId!,
                                      rating: _rating,
                                    );
                                    setState(() => _submitting = false);
                                    if (!mounted) return;
                                    ScaffoldMessenger.of(context).showSnackBar(
                                      SnackBar(content: Text(ok ? 'Voto inviato' : 'Impossibile inviare il voto')),
                                    );
                                  },
                            child: _submitting
                                ? const SizedBox(
                                    height: 16, width: 16,
                                    child: CircularProgressIndicator(strokeWidth: 2),
                                  )
                                : const Text('Invia voto'),
                          ),
                          const SizedBox(height: 40),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _infoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Row(
        children: [
          SizedBox(width: 120, child: Text('$label:', style: const TextStyle(color: Colors.white70))),
          Expanded(child: Text(value, style: const TextStyle(color: Colors.white))),
        ],
      ),
    );
  }
}
