package serialisation;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import java.io.IOException;
import java.io.PrintWriter;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;

/**
 *
 * @author tbouvard
 */
public class SerialisationLogin extends Serialisation {

    @Override
    public void serialiser(HttpServletRequest request, HttpServletResponse response) throws IOException {
        PrintWriter out = getWriterWithJsonHeader(response);
        JsonObject jsonContainer = new JsonObject();
        Boolean statut = (Boolean) request.getAttribute("statut");
        
        if (statut) {
            jsonContainer.addProperty("connexion", true);
            jsonContainer.addProperty("message", "Ok");
            
            HttpSession session = request.getSession();
            session.setAttribute("utilisateur", request.getParameter("login"));
            
            Gson gson = new GsonBuilder().setPrettyPrinting().create();
            String json = gson.toJson(jsonContainer);
            out.println(json);
        } else {
            jsonContainer.addProperty("connexion", false);
            jsonContainer.addProperty("message", "Mot de passe ou nom d'ultilisateur invalide.");
            Gson gson = new GsonBuilder().setPrettyPrinting().create();
            String json = gson.toJson(jsonContainer);
            out.println(json);
        }
    }
}
