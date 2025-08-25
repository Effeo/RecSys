import 'package:flutter/material.dart';

class ProfileAvatar extends StatelessWidget {
  final String label;
  final VoidCallback onTap;
  final bool isAddButton;
  final VoidCallback? onEdit;

  const ProfileAvatar({
    super.key,
    required this.label,
    required this.onTap,
    this.isAddButton = false,
    this.onEdit,
  });

  @override
  Widget build(BuildContext context) {
    final avatar = Container(
      width: 120,
      height: 120,
      decoration: BoxDecoration(
        color: isAddButton ? Colors.black54 : Colors.grey.shade800,
        borderRadius: BorderRadius.circular(8),
        image: isAddButton
            ? null
            : DecorationImage(
                // Placeholder “poster profile” astratto
                image: const AssetImage('assets/profile_placeholder.jpg'),
                fit: BoxFit.cover,
              ),
      ),
      child: isAddButton
          ? const Icon(Icons.add, size: 48, color: Colors.white70)
          : null,
    );

    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(8),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Stack(
            children: [
              avatar,
              if (!isAddButton && onEdit != null)
                Positioned(
                  right: 6,
                  top: 6,
                  child: InkWell(
                    onTap: onEdit,
                    child: Container(
                      padding: const EdgeInsets.all(6),
                      decoration: BoxDecoration(
                        color: Colors.black54,
                        borderRadius: BorderRadius.circular(999),
                      ),
                      child: const Icon(
                        Icons.edit,
                        size: 18,
                        color: Colors.white,
                      ),
                    ),
                  ),
                ),
            ],
          ),
          const SizedBox(height: 10),
          Text(
            label,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 16,
              letterSpacing: 0.3,
            ),
          ),
        ],
      ),
    );
  }
}
