import 'package:flutter/material.dart';
import 'package:rec_movies_frontend/utils/asset_picker.dart';

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
    final avatarBox = ClipRRect(
      borderRadius: BorderRadius.circular(8),
      child: Container(
        width: 120,
        height: 120,
        color: Colors.grey.shade800,
        alignment: Alignment.center,
        child: isAddButton
            ? const Icon(Icons.add, size: 48, color: Colors.white70)
            : Image.asset(
                AssetPicker.avatarForUser(label),
                width: 120,
                height: 120,
                fit: BoxFit.scaleDown, // << riduce se non entra
                errorBuilder: (ctx, _, __) => const _AvatarFallback(),
              ),
      ),
    );

    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(8),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Stack(
            children: [
              avatarBox,
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

class _AvatarFallback extends StatelessWidget {
  const _AvatarFallback();
  @override
  Widget build(BuildContext context) {
    return Container(
      width: 120,
      height: 120,
      color: Colors.black54,
      alignment: Alignment.center,
      child: const Icon(Icons.person, size: 42, color: Colors.white38),
    );
  }
}
