// lib/utils/asset_picker.dart
import '../models/movie.dart';

class AssetPicker {
  // Metti 5 poster a tua scelta in queste path (o rinominale come vuoi)
  static const List<String> posterAssets = [
    'assets/posters/poster1.jpg',
    'assets/posters/poster2.jpg',
    'assets/posters/poster3.jpg',
    'assets/posters/poster4.jpg',
    'assets/posters/poster5.jpg',
  ];

  // Metti 5 avatar profilo
  static const List<String> avatarAssets = [
    'assets/avatars/avatar1.jpg',
    'assets/avatars/avatar2.jpg',
    'assets/avatars/avatar3.jpg',
    'assets/avatars/avatar4.jpg',
    'assets/avatars/avatar5.jpg',
  ];

  /// Poster per film: “random” deterministico in base a id/titolo
  static String posterForMovie(Movie m) {
    final key = (m.movieId?.toString() ?? m.title).trim();
    final idx = _indexForKey(key, posterAssets.length);
    return posterAssets[idx];
  }

  /// Avatar per utente: deterministico in base allo userId/nome profilo
  static String avatarForUser(String userId) {
    final key = userId.trim();
    final idx = _indexForKey(key, avatarAssets.length);
    return avatarAssets[idx];
  }

  static int _indexForKey(String key, int modulo) {
    // hashCode può cambiare tra run; se vuoi totale stabilità, usa un CRC32/MD5.
    final h = key.hashCode & 0x7fffffff;
    return modulo == 0 ? 0 : (h % modulo);
  }
}
