import 'package:flutter/material.dart';
import 'package:rec_movies_frontend/utils/asset_picker.dart';
import '../models/movie.dart';

class MovieCard extends StatelessWidget {
  final Movie movie;
  final VoidCallback onTap;

  const MovieCard({super.key, required this.movie, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      child: SizedBox(
        width: 180,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Poster (16:9) con riduzione se non entra
            AspectRatio(
              aspectRatio: 16 / 9,
              child: ClipRRect(
                borderRadius: BorderRadius.circular(6),
                child: Container(
                  color: const Color(0xFF1B1B1B),
                  alignment: Alignment.center,
                  child: Image.asset(
                    AssetPicker.posterForMovie(movie),
                    fit:
                        BoxFit.scaleDown, // << riduce se troppo grande, no crop
                    errorBuilder: (ctx, _, __) => _PosterFallback(),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 8),
            Text(
              movie.title,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 14,
                fontWeight: FontWeight.w600,
              ),
            ),
            if (movie.score != null || movie.similarity != null)
              Text(
                movie.score != null
                    ? 'Score: ${movie.score!.toStringAsFixed(2)}'
                    : 'Sim: ${movie.similarity!.toStringAsFixed(2)}',
                style: const TextStyle(color: Colors.white70, fontSize: 12),
              ),
          ],
        ),
      ),
    );
  }
}

class _PosterFallback extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      color: const Color(0xFF2A2A2A),
      alignment: Alignment.center,
      child: const Icon(Icons.local_movies, color: Colors.white38, size: 28),
    );
  }
}
