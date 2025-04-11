{
  description = "Python 3.12 development environment for hf_transfer";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs = {
        nixpkgs.follows = "nixpkgs";
        # flake-utils.follows = "flake-utils";
      };
    };
  };
  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      rust-overlay,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };
        pythonVersions = [
          "39"
          "310"
          "311"
          "312"
        ];
        rustVersions = [
          "stable"
          # "nightly"
        ];

        mkPythonPackage =
          pythonVersion:
          let
            python = pkgs."python${pythonVersion}";
          in
          python.pkgs.buildPythonPackage {
            pname = "hf_transfer";
            version = "0.1.9-dev0";
            format = "pyproject";

            src = ./.;

            cargoDeps = pkgs.rustPlatform.importCargoLock {
              lockFile = ./Cargo.lock;
            };

            nativeBuildInputs = with pkgs; [
              rustPlatform.cargoSetupHook
              rustPlatform.maturinBuildHook
              pkg-config
              perl
              maturin
              python3Packages.setuptools
              python3Packages.wheel
            ];

            buildInputs = with pkgs; [
              openssl
            ];

            pythonImportsCheck = [ "hf_transfer" ];

            nativeCheckInputs = [ python.pkgs.pytest ];

            MATURIN_SETUP_ARGS = "--no-default-features";
          };

        pythonEnvs =
          builtins.mapAttrs
            (
              name: version:
              pkgs."python${version}".withPackages (ps: [
                (mkPythonPackage version)
                ps.pytest
                ps.exceptiongroup
              ])
            )
            (
              builtins.listToAttrs (
                map (v: {
                  name = "python${v}";
                  value = v;
                }) pythonVersions
              )
            );

        rustToolchains = builtins.listToAttrs (
          map (v: {
            name = "rust${v}";
            value = pkgs.rust-bin."${v}".latest.default;
          }) rustVersions
        );

        cargoLock = pkgs.rustPlatform.importCargoLock {
          lockFile = ./Cargo.lock;
        };
      in
      {
        packages = builtins.mapAttrs (name: version: mkPythonPackage version) (
          builtins.listToAttrs (
            map (v: {
              name = "python${v}";
              value = v;
            }) pythonVersions
          )
        );

        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            rustup
            python3Packages.venvShellHook
            perl
            pkg-config
            openssl
            maturin
          ];
          venvDir = "./.venv";
        };

        checks =
          let
            mkCheck =
              pythonVersion: rustVersion:
              let
                python = pythonEnvs."python${pythonVersion}";
              in
              pkgs.rustPlatform.buildRustPackage {
                pname = "hf_transfer-check";
                version = "0.1.9-dev0";
                src = self;
                cargoLock = {
                  lockFile = ./Cargo.lock;
                };

                nativeBuildInputs = [
                  python
                  rustToolchains."rust${rustVersion}"
                  pkgs.pkg-config
                  pkgs.openssl
                  pkgs.maturin
                  pkgs.perl
                  pkgs.python3Packages.setuptools
                  pkgs.python3Packages.wheel
                ];

                buildInputs = [ pkgs.openssl ];

                buildPhase = ''
                  # Run the checks
                  cargo clippy --all-targets --all-features -- -D warnings
                  cargo fmt --all -- --check
                  ${python}/bin/python -m pytest tests/ -v
                '';

                installPhase = "touch $out";
              };
          in
          builtins.foldl' (
            acc: pythonVersion:
            acc
            // builtins.foldl' (
              acc: rustVersion:
              acc
              // {
                "python${pythonVersion}-rust${rustVersion}" = mkCheck pythonVersion rustVersion;
              }
            ) { } rustVersions
          ) { } pythonVersions;
      }
    );
}
