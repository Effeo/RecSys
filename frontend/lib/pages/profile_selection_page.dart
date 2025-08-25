import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import '../services/api_client.dart';
import '../widgets/profile_avatar.dart';

class ProfileSelectionPage extends StatefulWidget {
  const ProfileSelectionPage({super.key});

  @override
  State<ProfileSelectionPage> createState() => _ProfileSelectionPageState();
}

class _ProfileSelectionPageState extends State<ProfileSelectionPage> {
  late Future<List<String>> _futureUsers;

  @override
  void initState() {
    super.initState();
    _futureUsers = ApiClient.fetchUserIds();
  }

  void _onOpenProfile(String userId) {
    context.push('/home/$userId');
  }

  void _onAddProfile() {
    // TODO: Naviga alla pagina di creazione profilo
    // context.push('/profiles/new');
    debugPrint('Add new profile');
  }

  void _onEditProfile(String userId) {
    // TODO: Naviga alla pagina di modifica profilo
    // context.push('/profiles/$userId/edit');
    debugPrint('Edit profile: $userId');
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF141414), // Netflix-like dark
      body: SafeArea(
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 1100),
            child: Column(
              children: [
                const SizedBox(height: 40),
                const Text(
                  'Chi vuole guardare?',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 36,
                    fontWeight: FontWeight.w700,
                    letterSpacing: 0.4,
                  ),
                ),
                const SizedBox(height: 40),
                FutureBuilder<List<String>>(
                  future: _futureUsers,
                  builder: (context, snapshot) {
                    if (snapshot.connectionState == ConnectionState.waiting) {
                      return const Padding(
                        padding: EdgeInsets.only(top: 80),
                        child: CircularProgressIndicator(),
                      );
                    }
                    if (snapshot.hasError) {
                      return Padding(
                        padding: const EdgeInsets.only(top: 80),
                        child: Text(
                          'Errore: ${snapshot.error}',
                          style: const TextStyle(color: Colors.redAccent),
                        ),
                      );
                    }

                    final users = snapshot.data ?? [];
                    final items = <Widget>[
                      ...users.map(
                        (u) => ProfileAvatar(
                          label: u,
                          onTap: () => _onOpenProfile(u),
                          onEdit: () => _onEditProfile(u),
                        ),
                      ),
                      ProfileAvatar(
                        label: 'Aggiungi profilo',
                        isAddButton: true,
                        onTap: _onAddProfile,
                      ),
                    ];

                    // --- CENTRATURA: Wrap centrato dentro uno scroll verticale ---
                    return Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 24),
                      child: LayoutBuilder(
                        builder: (context, constraints) {
                          // Spaziatura coerente con la UI
                          const horizontalGap = 28.0;
                          const verticalGap = 28.0;

                          return SingleChildScrollView(
                            child: Center(
                              child: Padding(
                                padding: const EdgeInsets.only(top: 30),
                                child: Wrap(
                                  spacing: horizontalGap,
                                  runSpacing: verticalGap,
                                  alignment: WrapAlignment.center,
                                  runAlignment: WrapAlignment.center,
                                  children: items,
                                ),
                              ),
                            ),
                          );
                        },
                      ),
                    );
                  },
                ),
                const SizedBox(height: 40),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
