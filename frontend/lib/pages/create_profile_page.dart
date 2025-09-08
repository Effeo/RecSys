import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import '../services/api_client.dart';

class CreateProfilePage extends StatefulWidget {
  const CreateProfilePage({super.key});

  @override
  State<CreateProfilePage> createState() => _CreateProfilePageState();
}

class _CreateProfilePageState extends State<CreateProfilePage> {
  final _formKey = GlobalKey<FormState>();

  // campi base
  final _userIdCtrl = TextEditingController();
  final _minYearCtrl = TextEditingController(text: '1990');
  final _preferredRuntimeCtrl = TextEditingController(text: '110');
  final _tolleranzaRuntimeCtrl = TextEditingController(text: '15');
  bool _preferAwardWinning = false;

  // multi-select generi
  final Set<String> _desiredGenres = {};
  final Set<String> _forbiddenGenres = {};

  // registi preferiti (semplice input con aggiunta manuale)
  final _directorCtrl = TextEditingController();
  final List<String> _favoriteDirectors = [];

  String? _submitError;
  bool _isSaving = false;

  // elenco generi usati nel backend
  static const List<String> _kGenres = [
    'Action',
    'Adventure',
    'Animation',
    'Children',
    'Comedy',
    'Crime',
    'Documentary',
    'Drama',
    'Fantasy',
    'Film_noir',
    'Horror',
    'Musical',
    'Mystery',
    'Romance',
    'Sci_fi',
    'Thriller',
    'War',
    'Western',
  ];

  @override
  void dispose() {
    _userIdCtrl.dispose();
    _minYearCtrl.dispose();
    _preferredRuntimeCtrl.dispose();
    _tolleranzaRuntimeCtrl.dispose();
    _directorCtrl.dispose();
    super.dispose();
  }

  void _addDirector() {
    final d = _directorCtrl.text.trim();
    if (d.isEmpty) return;
    if (!_favoriteDirectors.contains(d)) {
      setState(() => _favoriteDirectors.add(d));
    }
    _directorCtrl.clear();
  }

  Future<void> _onSubmit() async {
    setState(() => _submitError = null);
    if (!_formKey.currentState!.validate()) return;

    final userId = _userIdCtrl.text.trim();
    final minYear = int.tryParse(_minYearCtrl.text.trim());
    final prefRunText = _preferredRuntimeCtrl.text.trim();
    final prefRun = prefRunText.isEmpty ? null : int.tryParse(prefRunText);
    final tolRun = int.tryParse(_tolleranzaRuntimeCtrl.text.trim());

    final prefs = {
      "min_release_year": minYear,
      "generi_desiderati": _desiredGenres.toList(),
      "generi_vietati": _forbiddenGenres.toList(),
      "prefer_award_winning": _preferAwardWinning,
      "preferred_runtime": prefRun, // può essere null
      "tolleranza_runtime": tolRun,
      "favorite_directors": _favoriteDirectors,
    };

    setState(() => _isSaving = true);
    try {
      // usa POST /users/{user_id} del backend esistente
      await ApiClient.createOrUpdateUser(userId, prefs);
      if (!mounted) return;
      context.pop(true); // torna alla lista segnalando "creato"
    } catch (e) {
      setState(() => _submitError = e.toString());
    } finally {
      if (mounted) setState(() => _isSaving = false);
    }
  }

  Widget _buildGenreChips(String title, Set<String> model, {String? helper}) {
    return Card(
      color: const Color(0xFF1E1E1E),
      child: Padding(
        padding: const EdgeInsets.all(16), // più respiro
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              title,
              style: const TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.w600,
              ),
            ),
            if (helper != null)
              Padding(
                padding: const EdgeInsets.only(top: 6),
                child: Text(
                  helper,
                  style: const TextStyle(color: Colors.white70, fontSize: 12.5),
                ),
              ),
            const SizedBox(height: 12),
            Wrap(
              spacing: 12, // <— più spazio orizzontale
              runSpacing:
                  12, // <— più spazio verticale (niente valori negativi)
              children: _kGenres.map((g) {
                final selected = model.contains(g);
                return FilterChip(
                  label: Text(g),
                  selected: selected,
                  onSelected: (v) {
                    setState(() {
                      if (v) {
                        // opzionale: impedisci conflitto desiderati/vietati
                        // if (model == _desiredGenres) _forbiddenGenres.remove(g);
                        // else _desiredGenres.remove(g);
                        model.add(g);
                      } else {
                        model.remove(g);
                      }
                    });
                  },
                );
              }).toList(),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final isWide = MediaQuery.of(context).size.width > 900;

    return Scaffold(
      backgroundColor: const Color(0xFF141414),
      appBar: AppBar(
        backgroundColor: const Color(0xFF141414),
        elevation: 0,
        title: const Text('Crea nuovo profilo'),
      ),
      body: SafeArea(
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 1100),
            child: Form(
              key: _formKey,
              child: ListView(
                padding: const EdgeInsets.all(20),
                children: [
                  // riga: nome + min year
                  Flex(
                    direction: isWide ? Axis.horizontal : Axis.vertical,
                    children: [
                      Expanded(
                        child: TextFormField(
                          controller: _userIdCtrl,
                          style: const TextStyle(color: Colors.white),
                          decoration: const InputDecoration(
                            labelText: 'Nome profilo',
                            labelStyle: TextStyle(color: Colors.white70),
                            filled: true,
                            fillColor: Color(0xFF1E1E1E),
                          ),
                          validator: (v) => (v == null || v.trim().isEmpty)
                              ? 'Inserisci un nome profilo'
                              : null,
                        ),
                      ),
                      const SizedBox(width: 16, height: 16),
                      Expanded(
                        child: TextFormField(
                          controller: _minYearCtrl,
                          keyboardType: TextInputType.number,
                          style: const TextStyle(color: Colors.white),
                          decoration: const InputDecoration(
                            labelText: 'Anno minimo uscita (es. 1990)',
                            labelStyle: TextStyle(color: Colors.white70),
                            filled: true,
                            fillColor: Color(0xFF1E1E1E),
                          ),
                          validator: (v) {
                            if (v == null || v.trim().isEmpty) return null;
                            final n = int.tryParse(v);
                            if (n == null || n < 1900 || n > 2025) {
                              return 'Inserisci un anno valido';
                            }
                            return null;
                          },
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 20),

                  // riga: runtime + tolleranza + premi
                  Flex(
                    direction: isWide ? Axis.horizontal : Axis.vertical,
                    children: [
                      Expanded(
                        child: TextFormField(
                          controller: _preferredRuntimeCtrl,
                          keyboardType: TextInputType.number,
                          style: const TextStyle(color: Colors.white),
                          decoration: const InputDecoration(
                            labelText: 'Durata preferita (min) — opzionale',
                            labelStyle: TextStyle(color: Colors.white70),
                            filled: true,
                            fillColor: Color(0xFF1E1E1E),
                          ),
                        ),
                      ),
                      const SizedBox(width: 16, height: 16),
                      Expanded(
                        child: TextFormField(
                          controller: _tolleranzaRuntimeCtrl,
                          keyboardType: TextInputType.number,
                          style: const TextStyle(color: Colors.white),
                          decoration: const InputDecoration(
                            labelText: 'Tolleranza durata (min)',
                            labelStyle: TextStyle(color: Colors.white70),
                            filled: true,
                            fillColor: Color(0xFF1E1E1E),
                          ),
                          validator: (v) {
                            final n = int.tryParse(v ?? '');
                            if (n == null || n < 0 || n > 180) {
                              return '0–180';
                            }
                            return null;
                          },
                        ),
                      ),
                      const SizedBox(width: 16, height: 16),
                      Expanded(
                        child: SwitchListTile(
                          title: const Text(
                            'Preferisci film premiati',
                            style: TextStyle(color: Colors.white),
                          ),
                          value: _preferAwardWinning,
                          onChanged: (v) =>
                              setState(() => _preferAwardWinning = v),
                          contentPadding: EdgeInsets.zero,
                        ),
                      ),
                    ],
                  ),

                  const SizedBox(height: 20),
                  _buildGenreChips(
                    'Generi desiderati',
                    _desiredGenres,
                    helper: 'Seleziona i generi che vuoi vedere',
                  ),
                  const SizedBox(height: 16),
                  _buildGenreChips(
                    'Generi vietati',
                    _forbiddenGenres,
                    helper: 'Questi generi verranno esclusi',
                  ),

                  const SizedBox(height: 16),
                  // registi preferiti
                  Card(
                    color: const Color(0xFF1E1E1E),
                    child: Padding(
                      padding: const EdgeInsets.all(16), // più respiro
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            'Registi preferiti (opzionale)',
                            style: TextStyle(
                              color: Colors.white,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                          const SizedBox(height: 10),
                          Row(
                            children: [
                              Expanded(
                                child: TextField(
                                  controller: _directorCtrl,
                                  style: const TextStyle(color: Colors.white),
                                  decoration: const InputDecoration(
                                    hintText:
                                        'Aggiungi regista e premi invio/➕',
                                    hintStyle: TextStyle(color: Colors.white54),
                                    filled: true,
                                    fillColor: Color(0xFF2A2A2A),
                                  ),
                                  onSubmitted: (_) => _addDirector(),
                                ),
                              ),
                              const SizedBox(width: 8),
                              IconButton(
                                onPressed: _addDirector,
                                icon: const Icon(
                                  Icons.add,
                                  color: Colors.white70,
                                ),
                                tooltip: 'Aggiungi',
                              ),
                            ],
                          ),
                          const SizedBox(height: 10),
                          Wrap(
                            spacing: 8,
                            runSpacing: 8,
                            children: _favoriteDirectors
                                .map(
                                  (d) => Chip(
                                    label: Text(d),
                                    onDeleted: () => setState(
                                      () => _favoriteDirectors.remove(d),
                                    ),
                                  ),
                                )
                                .toList(),
                          ),
                        ],
                      ),
                    ),
                  ),

                  if (_submitError != null) ...[
                    const SizedBox(height: 10),
                    Text(
                      _submitError!,
                      style: const TextStyle(
                        color: Colors.redAccent,
                        fontSize: 13,
                      ),
                    ),
                  ],

                  const SizedBox(height: 16),
                  Row(
                    children: [
                      ElevatedButton.icon(
                        onPressed: _isSaving ? null : _onSubmit,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.green.shade600, // <— VERDE
                          foregroundColor: Colors.white,
                          disabledBackgroundColor: Colors.green.shade300,
                        ),
                        icon: _isSaving
                            ? const SizedBox(
                                width: 16,
                                height: 16,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2,
                                  valueColor: AlwaysStoppedAnimation(
                                    Colors.white,
                                  ),
                                ),
                              )
                            : const Icon(Icons.check),
                        label: const Text('Crea profilo'),
                      ),
                      const SizedBox(width: 12),
                      TextButton(
                        onPressed: _isSaving ? null : () => context.pop(false),
                        child: const Text('Annulla'),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
