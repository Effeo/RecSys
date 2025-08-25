import 'package:flutter/material.dart';
import '../models/movie.dart';
import 'movie_card.dart';

class SectionRow extends StatelessWidget {
  final String title;
  final List<Movie> movies;
  final void Function(Movie) onTapMovie;

  const SectionRow({
    super.key,
    required this.title,
    required this.movies,
    required this.onTapMovie,
  });

  @override
  Widget build(BuildContext context) {
    if (movies.isEmpty) {
      return Padding(
        padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 24),
        child: Text(
          '$title • Nessuna corrispondenza',
          style: const TextStyle(color: Colors.white60, fontSize: 16),
        ),
      );
    }

    return Padding(
      padding: const EdgeInsets.only(top: 10, bottom: 24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 24),
            child: Text(
              title,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 20,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
          const SizedBox(height: 10),
          SizedBox(
            height: 180 * (9/16) + 50, // altezza poster + testo
            child: ListView.separated(
              padding: const EdgeInsets.symmetric(horizontal: 24),
              scrollDirection: Axis.horizontal,
              itemBuilder: (_, i) => MovieCard(
                movie: movies[i],
                onTap: () => onTapMovie(movies[i]),
              ),
              separatorBuilder: (_, __) => const SizedBox(width: 14),
              itemCount: movies.length,
            ),
          ),
        ],
      ),
    );
  }
}
