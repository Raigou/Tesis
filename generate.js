import { readdir, stat } from "fs/promises";
import { join } from "path";

const IGNORE_FOLDERS = ["node_modules", ".git", "dist", "build", "entorno_gpu"];

async function listarEstructura(dir, nivel = 0) {
    try {
        const archivos = await readdir(dir);
        
        for (const archivo of archivos) {
            const rutaCompleta = join(dir, archivo);
            const info = await stat(rutaCompleta);
            
            if (IGNORE_FOLDERS.includes(archivo)) continue; // Ignorar carpetas específicas

            console.log(" ".repeat(nivel * 2) + "|- " + archivo);
            
            if (info.isDirectory()) {
                await listarEstructura(rutaCompleta, nivel + 1);
            }
        }
    } catch (error) {
        console.error(`Error al leer el directorio ${dir}:`, error);
    }
}

// Ejecutar el script en el directorio actual
console.log("Estructura del proyecto:");
listarEstructura(process.cwd());