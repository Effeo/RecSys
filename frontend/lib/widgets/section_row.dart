import 'package:flutter/material.dart';
import 'package:rec_movies_frontend/utils/asset_picker.dart';
import '../models/movie.dart';

class SectionRow extends StatefulWidget {
  final String title;
  final List<Movie> movies;
  final void Function(Movie) onTapMovie;

  // opzionali / stile
  final TextStyle? titleStyle;
  final Color accentColor;
  final bool showArrows;
  final bool showExploreBadge;

  const SectionRow({
    super.key,
    required this.title,
    required this.movies,
    required this.onTapMovie,
    this.titleStyle,
    this.accentColor = const Color(0xFFE50914),
    this.showArrows = true,
    this.showExploreBadge = false,
  });

  @override
  State<SectionRow> createState() => _SectionRowState();
}

class _SectionRowState extends State<SectionRow> {
  final _controller = ScrollController();
  bool _canScrollBack = false;
  bool _canScrollForward = false;

  @override
  void initState() {
    super.initState();
    _controller.addListener(_onScroll);
    WidgetsBinding.instance.addPostFrameCallback((_) => _updateArrows());
  }

  @override
  void dispose() {
    _controller.removeListener(_onScroll);
    _controller.dispose();
    super.dispose();
  }

  void _onScroll() => _updateArrows();

  void _updateArrows() {
    if (!_controller.hasClients) return;
    final maxScroll = _controller.position.maxScrollExtent;
    final offset = _controller.offset;
    final canBack = offset > 0;
    final canFwd = offset < maxScroll - 2;
    if (canBack != _canScrollBack || canFwd != _canScrollForward) {
      setState(() {
        _canScrollBack = canBack;
        _canScrollForward = canFwd;
      });
    }
  }

  Future<void> _pageScroll(bool forward) async {
    if (!_controller.hasClients) return;
    final viewport = _controller.position.viewportDimension;
    final delta = viewport * 0.7; // “pagina” ~70% viewport
    final target = (_controller.offset + (forward ? delta : -delta)).clamp(
      0,
      _controller.position.maxScrollExtent,
    );
    await _controller.animateTo(
      target.toDouble(),
      duration: const Duration(milliseconds: 320),
      curve: Curves.easeOutCubic,
    );
  }

  @override
  Widget build(BuildContext context) {
    if (widget.movies.isEmpty) {
      return const SizedBox.shrink();
    }

    return Padding(
      padding: const EdgeInsets.only(bottom: 18),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Titolo sezione
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 24),
            child: Row(
              children: [
                Text(
                  widget.title,
                  style:
                      widget.titleStyle ??
                      const TextStyle(
                        color: Colors.white,
                        fontSize: 20,
                        fontWeight: FontWeight.w800,
                      ),
                ),
                const SizedBox(width: 8),
                Container(
                  width: 6,
                  height: 6,
                  decoration: BoxDecoration(
                    color: widget.accentColor,
                    shape: BoxShape.circle,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 10),

          // RIGA con overlay bottoni + fade
          SizedBox(
            height: 210, // altezza card
            child: Stack(
              children: [
                // Lista orizzontale
                ListView.separated(
                  controller: _controller,
                  padding: const EdgeInsets.symmetric(horizontal: 48),
                  scrollDirection: Axis.horizontal,
                  itemCount: widget.movies.length,
                  separatorBuilder: (_, __) => const SizedBox(width: 12),
                  itemBuilder: (context, i) {
                    final m = widget.movies[i];
                    return _MovieCard(
                      movie: m,
                      onTap: () => widget.onTapMovie(m),
                      showExploreBadge: widget.showExploreBadge,
                    );
                  },
                ),

                // fade sinistra
                Positioned.fill(
                  left: 0,
                  child: IgnorePointer(
                    ignoring: true,
                    child: Container(
                      decoration: const BoxDecoration(
                        gradient: LinearGradient(
                          begin: Alignment.centerLeft,
                          end: Alignment.centerRight,
                          colors: [Color(0xFF141414), Colors.transparent],
                          stops: [0.0, 0.12],
                        ),
                      ),
                    ),
                  ),
                ),
                // fade destra
                Positioned.fill(
                  right: 0,
                  child: IgnorePointer(
                    ignoring: true,
                    child: Container(
                      decoration: const BoxDecoration(
                        gradient: LinearGradient(
                          begin: Alignment.centerRight,
                          end: Alignment.centerLeft,
                          colors: [Color(0xFF141414), Colors.transparent],
                          stops: [0.0, 0.12],
                        ),
                      ),
                    ),
                  ),
                ),

                // Frecce
                if (widget.showArrows) ...[
                  // sinistra
                  Positioned(
                    left: 6,
                    top: 0,
                    bottom: 0,
                    child: _ArrowButton(
                      enabled: _canScrollBack,
                      icon: Icons.chevron_left,
                      onTap: () => _pageScroll(false),
                    ),
                  ),
                  // destra
                  Positioned(
                    right: 6,
                    top: 0,
                    bottom: 0,
                    child: _ArrowButton(
                      enabled: _canScrollForward,
                      icon: Icons.chevron_right,
                      onTap: () => _pageScroll(true),
                    ),
                  ),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _ArrowButton extends StatelessWidget {
  final bool enabled;
  final IconData icon;
  final VoidCallback onTap;
  const _ArrowButton({
    required this.enabled,
    required this.icon,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Opacity(
      opacity: enabled ? 1 : 0.0,
      child: InkWell(
        onTap: enabled ? onTap : null,
        borderRadius: BorderRadius.circular(28),
        child: Container(
          width: 42,
          height: 42,
          decoration: BoxDecoration(
            color: const Color(0xFF2A2A2A).withOpacity(0.9),
            shape: BoxShape.circle,
            border: Border.all(color: Colors.white10),
            boxShadow: const [
              BoxShadow(color: Colors.black54, blurRadius: 8, spreadRadius: 2),
            ],
          ),
          child: Icon(icon, color: Colors.white, size: 28),
        ),
      ),
    );
  }
}

class _PosterFallbackMini extends StatelessWidget {
  const _PosterFallbackMini();
  @override
  Widget build(BuildContext context) {
    return Container(
      color: const Color(0xFF2A2A2A),
      alignment: Alignment.center,
      child: const Icon(Icons.local_movies, color: Colors.white38, size: 20),
    );
  }
}

class _MovieCard extends StatelessWidget {
  final Movie movie;
  final VoidCallback onTap;
  final bool showExploreBadge;
  const _MovieCard({
    required this.movie,
    required this.onTap,
    required this.showExploreBadge,
  });

  @override
  Widget build(BuildContext context) {
    // dimensioni card stile “locandina”
    const w = 140.0;
    const h = 210.0;

    final isExplore = (movie.pickStrategy == 'explore');
    final title = movie.title;
    final subtitleParts = <String>[];
    if (movie.score != null) {
      subtitleParts.add('Score ${_fmt(movie.score)}');
    } else if (movie.similarity != null) {
      subtitleParts.add('Sim ${_fmt(movie.similarity)}');
    }
    if (movie.releaseDate != null) {
      final year = movie.releaseDate!.split('-').first;
      subtitleParts.add(year);
    }
    final subtitle = subtitleParts.join(' · ');

    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(10),
      child: Container(
        width: w,
        height: h,
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(10),
          color: const Color(0xFF1B1B1B),
          boxShadow: const [
            BoxShadow(color: Colors.black54, blurRadius: 10, spreadRadius: 1),
          ],
        ),
        clipBehavior: Clip.antiAlias,
        child: Stack(
          children: [
            // Poster placeholder (sostituisci con le tue immagini se le hai)
            Positioned.fill(
              child: Container(
                color: const Color(0xFF1B1B1B),
                alignment: Alignment.center,
                child: Image.asset(
                  AssetPicker.posterForMovie(movie),
                  fit: BoxFit.scaleDown, // << riduce se necessario
                  errorBuilder: (ctx, _, __) => const _PosterFallbackMini(),
                ),
              ),
            ),
            // gradient bottom per leggibilità testo
            Positioned(
              left: 0,
              right: 0,
              bottom: 0,
              height: 86,
              child: Container(
                decoration: const BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.bottomCenter,
                    end: Alignment.topCenter,
                    colors: [Color(0xCC000000), Colors.transparent],
                  ),
                ),
              ),
            ),
            // badge NOVITÀ (solo explore)
            if (showExploreBadge && isExplore)
              Positioned(
                top: 8,
                left: 8,
                child: Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 8,
                    vertical: 4,
                  ),
                  decoration: BoxDecoration(
                    color: const Color(0xFFE50914),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: const Text(
                    'NOVITÀ',
                    style: TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.w900,
                      fontSize: 11,
                    ),
                  ),
                ),
              ),
            // testo
            Positioned(
              left: 8,
              right: 8,
              bottom: 8,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.w800,
                      fontSize: 13.5,
                      height: 1.15,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    subtitle,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(
                      color: Colors.white70,
                      fontSize: 11.5,
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

  String _fmt(num? v) {
    if (v == null) return '';
    final d = (v is double) ? v : v.toDouble();
    final rounded = (d * 100).roundToDouble() / 100.0;
    return rounded.toStringAsFixed(2);
  }
}
