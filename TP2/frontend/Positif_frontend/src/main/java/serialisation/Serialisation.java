package serialisation;

import java.io.IOException;
import java.io.PrintWriter;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

/**
 *
 * @author tbouvard
 */
public abstract class Serialisation {
    
    protected PrintWriter getWriterWithJsonHeader(HttpServletResponse response) throws IOException{
        response.setContentType("application/json;charset=UTF-8");
        return response.getWriter();
    }
    
    public abstract void serialiser(HttpServletRequest request, HttpServletResponse response) throws IOException;
}
