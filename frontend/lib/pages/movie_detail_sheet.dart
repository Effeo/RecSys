import 'package:flutter/material.dart';
import 'package:rec_movies_frontend/utils/asset_picker.dart';
import '../models/movie.dart';
import '../services/api_client.dart';

class MovieDetailSheet extends StatefulWidget {
  final Movie movie;
  final String userId;

  const MovieDetailSheet({
    super.key,
    required this.movie,
    required this.userId,
  });

  static Future<void> show(BuildContext context, Movie movie, String userId) {
    return showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: const Color(0xFF0F0F0F),
      builder: (_) => FractionallySizedBox(
        heightFactor: 0.92,
        child: MovieDetailSheet(movie: movie, userId: userId),
      ),
    );
  }

  @override
  State<MovieDetailSheet> createState() => _MovieDetailSheetState();
}

class _DetailPosterFallback extends StatelessWidget {
  const _DetailPosterFallback();
  @override
  Widget build(BuildContext context) {
    return Container(
      color: const Color(0xFF2A2A2A),
      alignment: Alignment.center,
      child: const Icon(Icons.local_movies, color: Colors.white38, size: 36),
    );
  }
}

class _MovieDetailSheetState extends State<MovieDetailSheet> {
  int _rating = 0;
  bool _submitting = false;

  @override
  Widget build(BuildContext context) {
    final m = widget.movie;

    return SafeArea(
      child: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Color(0xFF0F0F0F), Color(0xFF141414)],
          ),
        ),
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Header
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Expanded(
                    child: Text(
                      m.title,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.w900,
                        color: Colors.white,
                        letterSpacing: 0.1,
                      ),
                    ),
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
                    ClipRRect(
                      borderRadius: BorderRadius.circular(10),
                      child: Container(
                        color: const Color(0xFF1B1B1B),
                        width: 280,
                        height: 280 * 9 / 16,
                        alignment: Alignment.center,
                        child: Image.asset(
                          AssetPicker.posterForMovie(m),
                          fit: BoxFit.scaleDown, // << riduce se non entra
                          errorBuilder: (context, _, __) =>
                              const _DetailPosterFallback(),
                        ),
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
                            _infoRow(
                              'Durata',
                              (m.runtime?.toString() ?? 'n/d'),
                            ),
                            if (m.awards != null)
                              _infoRow('Premi', '${m.awards}'),
                            if (m.score != null)
                              _infoRow('Score', _fmt(m.score ?? 0)),
                            if (m.similarity != null)
                              _infoRow('Similarità', _fmt(m.similarity ?? 0)),
                            if (m.imdbUrl != null)
                              Padding(
                                padding: const EdgeInsets.only(top: 8),
                                child: SelectableText(
                                  m.imdbUrl!,
                                  style: const TextStyle(
                                    color: Colors.lightBlueAccent,
                                  ),
                                ),
                              ),
                            const SizedBox(height: 24),

                            // Rating
                            const Text(
                              'Valuta questo titolo',
                              style: TextStyle(
                                color: Colors.white,
                                fontSize: 16,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                            const SizedBox(height: 8),
                            Row(
                              children: List.generate(5, (i) {
                                final idx = i + 1;
                                final filled = _rating >= idx;
                                return IconButton(
                                  onPressed: () =>
                                      setState(() => _rating = idx),
                                  icon: Icon(
                                    filled ? Icons.star : Icons.star_border,
                                    color: filled
                                        ? Colors.amber
                                        : Colors.white70,
                                    size: 28,
                                  ),
                                );
                              }),
                            ),
                            const SizedBox(height: 12),
                            SizedBox(
                              height: 44,
                              child: ElevatedButton(
                                style: ElevatedButton.styleFrom(
                                  backgroundColor: const Color(0xFFE50914),
                                  foregroundColor: Colors.white,
                                  shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(10),
                                  ),
                                ),
                                onPressed:
                                    (_rating == 0 ||
                                        m.movieId == null ||
                                        _submitting)
                                    ? null
                                    : () async {
                                        setState(() => _submitting = true);
                                        final ok = await ApiClient.submitRating(
                                          userId: widget.userId,
                                          movieId: m.movieId!,
                                          rating: _rating,
                                        );
                                        if (!mounted) return;
                                        setState(() => _submitting = false);
                                        ScaffoldMessenger.of(
                                          context,
                                        ).showSnackBar(
                                          SnackBar(
                                            content: Text(
                                              ok
                                                  ? 'Voto inviato'
                                                  : 'Impossibile inviare il voto',
                                            ),
                                            backgroundColor: ok
                                                ? Colors.green
                                                : Colors.red,
                                          ),
                                        );
                                      },
                                child: _submitting
                                    ? const SizedBox(
                                        height: 18,
                                        width: 18,
                                        child: CircularProgressIndicator(
                                          strokeWidth: 2,
                                          color: Colors.white,
                                        ),
                                      )
                                    : const Text(
                                        'Invia voto',
                                        style: TextStyle(
                                          fontWeight: FontWeight.w800,
                                          letterSpacing: 0.2,
                                        ),
                                      ),
                              ),
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
      ),
    );
  }

  Widget _infoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Row(
        children: [
          SizedBox(
            width: 120,
            child: Text(
              '$label:',
              style: const TextStyle(
                color: Colors.white70,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
          Expanded(
            child: Text(value, style: const TextStyle(color: Colors.white)),
          ),
        ],
      ),
    );
  }

  String _fmt(num v) => (v is double ? v : v.toDouble()).toStringAsFixed(2);
}
